"""Validator replay HTTP API.

Endpoints (deliberately RPC-style, since real chain interaction is mocked):
- `POST /sample`     — submit a list of receipts + epoch_seed → returns sampled set.
- `POST /replay`     — replay sampled receipts; returns verdicts.
- `POST /run_epoch`  — convenience: sample + replay + (mock) slashing call in one shot.
- `GET  /healthz`
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from collections.abc import Callable
from typing import Annotated, Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from mining_types import Receipt, SlashingEvidence
from mining_types.crypto import sha256_hex, verify_ed25519
from pydantic import BaseModel, Field

from validator_replay.chain import ChainRpcClient, ChainUnreachableError, OperatorStake
from validator_replay.config import ValidatorConfig
from validator_replay.registry import OperatorRegistry, allow_unsigned_ingest
from validator_replay.replay import ReplayResult, ReplayVerdict, replay_receipt
from validator_replay.sampler import CommitReveal, StakeWeightedSampler
from validator_replay.weights import (
    compute_weight_vector,
    encode_submit_weights_call,
)
from validator_replay.weights import (
    submit_weights as submit_yuma_weights,
)

logger = logging.getLogger(__name__)
_BEARER = HTTPBearer(auto_error=False)
_REPLAY_INPUT_PATHS = {"/replay", "/run_epoch"}


def _is_production() -> bool:
    return os.environ.get("OROGEN_ENV", "").lower() == "production"


def _expected_api_token() -> str:
    return (
        os.environ.get("VALIDATOR_API_TOKEN", "")
        or os.environ.get("INTERNAL_AUTH_TOKEN", "")
    ).strip()


async def require_validator_auth(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)] = None,
) -> None:
    expected = _expected_api_token()
    if not expected:
        if _is_production():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="validator api token not configured",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return
    if creds is None or creds.scheme.lower() != "bearer" or creds.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="validator auth required",
            headers={"WWW-Authenticate": "Bearer"},
        )


class SampleRequest(BaseModel):
    receipts: list[Receipt]
    epoch_seed: str
    operator_stake: dict[str, int] = Field(default_factory=dict)


class ReplayRequest(BaseModel):
    receipts: list[Receipt]
    replay_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RunEpochRequest(BaseModel):
    receipts: list[Receipt]
    epoch_seed: str
    operator_stake: dict[str, int] = Field(default_factory=dict)
    replay_inputs: dict[str, dict[str, Any]] = Field(default_factory=dict)


class SubmitWeightsRequest(BaseModel):
    """Compute + sign + submit a Yuma weight vector for one epoch.

    Reward-crank #3: the validator's per-epoch quality vote. `epoch` is the
    Yuma epoch label (u64). `faulty_operators` is the set of operator_ids
    (hex AccountIds) that produced a MISMATCH this epoch — these get weight 0.
    When `faulty_operators` is empty (the bootstrap / no-fault case), every
    active operator gets `u16::MAX`, matching the `drive-epoch0.cjs` stand-in.
    """

    epoch: int
    faulty_operators: set[str] = Field(default_factory=set)
    equal_weight: bool = True
    operator_stake: dict[str, int] = Field(default_factory=dict)


def _json_size_bytes(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


class ReplayInputBodyLimitMiddleware:
    """ASGI request-body limiter for routes that can carry replay inputs."""

    def __init__(
        self,
        app: Callable[[dict[str, Any], Callable[..., Any], Callable[..., Any]], Any],
        *,
        max_bytes: int,
    ) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None:
        if scope.get("type") != "http" or scope.get("path") not in _REPLAY_INPUT_PATHS:
            await self.app(scope, receive, send)
            return

        headers = {
            key.lower(): value
            for key, value in scope.get("headers", [])
        }
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._send_413(send)
                    return
            except ValueError:
                pass

        seen = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal seen
            message = await receive()
            if message.get("type") == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise ReplayInputTooLarge
            return message

        async def tracking_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except ReplayInputTooLarge:
            if not response_started:
                await self._send_413(send)

    async def _send_413(self, send: Callable[[dict[str, Any]], Any]) -> None:
        body = json.dumps({
            "detail": (
                "replay request body too large for configured ephemeral policy "
                f"(max {self.max_bytes} bytes)"
            )
        }).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body})


class ReplayInputTooLarge(Exception):
    pass


async def _replay_receipt_async(receipt: Receipt, **kwargs: Any) -> ReplayResult:
    return await asyncio.to_thread(replay_receipt, receipt, **kwargs)


def build_app(
    config: ValidatorConfig,
    registry: OperatorRegistry | None = None,
) -> FastAPI:
    config.validate_replay_input_policy(production=_is_production())
    app = FastAPI(title="validator-replay", version="0.1.0")
    app.add_middleware(
        ReplayInputBodyLimitMiddleware,
        max_bytes=config.replay_input_max_bytes,
    )
    app.state.config = config
    app.state.chain_client = ChainRpcClient(config.resolved_chain_rpc_url())
    reg = registry or OperatorRegistry.from_env()
    app.state.registry = reg

    def _verify_receipts(receipts: list[Receipt]) -> tuple[list[Receipt], int, int]:
        """N-W-01: split receipts into (accepted, rejected_unknown, rejected_invalid).

        Receipts whose operator_id is not registered (unknown pubkey) or
        whose `operator_signature` does not verify are dropped. The counts
        are surfaced in the response so observers can correlate spikes.
        """
        if allow_unsigned_ingest():
            return list(receipts), 0, 0
        accepted: list[Receipt] = []
        unknown = 0
        invalid = 0
        for r in receipts:
            pub = reg.get(r.operator_id)
            if pub is None:
                unknown += 1
                logger.warning(
                    "rejecting receipt %s: operator_id %r not in registry",
                    r.job_id, r.operator_id,
                )
                continue
            if not verify_ed25519(pub, r.signing_payload(), r.operator_signature):
                invalid += 1
                logger.warning(
                    "rejecting receipt %s: invalid operator_signature for %r",
                    r.job_id, r.operator_id,
                )
                continue
            accepted.append(r)
        return accepted, unknown, invalid
    submitted: list[SlashingEvidence] = []
    app.state.slashing_submitted = submitted
    # M-04: commit-reveal salt is per-epoch, not per-call. The salt is
    # generated once per `epoch_seed` and cached so subsequent /sample
    # calls return the same sampled set (or refuse to re-roll). This
    # prevents a validator from re-shuffling until they find a favourable
    # sample before committing.
    salts_by_epoch: dict[str, str] = {}
    app.state.salts_by_epoch = salts_by_epoch
    # M-02: track outbound HTTP status codes from the slashing-evidence
    # secondary sink so /healthz can surface delivery failures.
    sink_status_counts: dict[str, int] = {}
    app.state.sink_status_counts = sink_status_counts

    def _get_or_create_salt(epoch_seed: str) -> str:
        existing = salts_by_epoch.get(epoch_seed)
        if existing is not None:
            return existing
        new_salt = secrets.token_hex(16)
        salts_by_epoch[epoch_seed] = new_salt
        return new_salt

    def _enforce_replay_input_policy(replay_inputs: dict[str, dict[str, Any]]) -> None:
        """Apply the replay-input privacy boundary before worker replay.

        The validator needs original request inputs to independently recompute a
        response, but this service must not become a durable prompt store. The
        implemented policy is deliberately narrow: bound request size, keep the
        values only in request-local memory, and never include them in responses
        or logs.
        """
        if not replay_inputs:
            return
        size = _json_size_bytes(replay_inputs)
        if size > config.replay_input_max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "replay_inputs too large for configured ephemeral policy "
                    f"({size} > {config.replay_input_max_bytes} bytes)"
                ),
            )

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "validator_id": config.validator_id,
            "chain_rpc_url": config.resolved_chain_rpc_url(),
            "replay_input_policy": config.resolved_replay_input_policy(),
            "replay_input_max_bytes": config.replay_input_max_bytes,
            "sink_status_counts": dict(sink_status_counts),
        }

    @app.get("/internal/operators", dependencies=[Depends(require_validator_auth)])
    async def operators() -> dict[str, Any]:
        """Surface the chain-derived operator set + stakes.

        C-04: on chain-RPC failure this RAISES (returns 503) unless stub
        fallback was opted into. We no longer silently substitute stub data.
        """
        try:
            ops = app.state.chain_client.get_operator_stakes()
        except ChainUnreachableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "rpc_url": config.resolved_chain_rpc_url(),
            "operators": [
                {"operator_id": o.operator_id, "stake": o.stake, "active": o.active}
                for o in ops
            ],
        }

    @app.post("/sample", dependencies=[Depends(require_validator_auth)])
    async def sample(req: SampleRequest) -> dict[str, Any]:
        # N-W-01: drop receipts with unknown / invalid operator signatures
        # BEFORE sampling, so the sampled set can't include forgeries.
        accepted, unknown, invalid = _verify_receipts(req.receipts)
        sampler = StakeWeightedSampler(
            sample_rate=config.sample_rate, operator_stake=req.operator_stake,
        )
        commit = CommitReveal(
            epoch_seed=req.epoch_seed,
            validator_id=config.validator_id,
            salt=_get_or_create_salt(req.epoch_seed),
        )
        chosen = sampler.sample(accepted, commit)
        return {
            "sampled": [r.model_dump(mode="json") for r in chosen],
            "commit": commit.commit(),
            "rejected_unknown_operator": unknown,
            "rejected_invalid_signature": invalid,
        }

    @app.post("/replay", dependencies=[Depends(require_validator_auth)])
    async def replay(req: ReplayRequest) -> dict[str, Any]:
        _enforce_replay_input_policy(req.replay_inputs)
        # N-W-01: refuse to act on forged receipts. Unknown / invalid
        # signatures are dropped before replay.
        accepted, unknown, invalid = _verify_receipts(req.receipts)
        results: list[dict[str, Any]] = []
        worker_url = config.worker_replay_url or None
        for r in accepted:
            res = await _replay_receipt_async(
                r,
                worker_url=worker_url,
                replay_input=req.replay_inputs.get(r.job_id),
            )
            results.append(
                {
                    "job_id": r.job_id,
                    "operator_id": r.operator_id,
                    "verdict": res.verdict.value,
                    "fault": res.fault.value if res.fault else None,
                    "detail": res.detail,
                }
            )
        return {
            "results": results,
            "rejected_unknown_operator": unknown,
            "rejected_invalid_signature": invalid,
        }

    @app.post("/run_epoch", dependencies=[Depends(require_validator_auth)])
    async def run_epoch(req: RunEpochRequest) -> dict[str, Any]:
        _enforce_replay_input_policy(req.replay_inputs)
        # N-W-01: refuse to act on forged receipts. Slashing evidence
        # generated from a forged receipt would let an attacker grief any
        # operator simply by submitting a hand-crafted Receipt.
        accepted, unknown, invalid = _verify_receipts(req.receipts)
        # If the caller did not pass an explicit `operator_stake`, pull it
        # from the chain. C-04: this RAISES on RPC failure unless stub
        # fallback was explicitly opted into.
        operator_stake = req.operator_stake
        if not operator_stake:
            try:
                operator_stake = {
                    o.operator_id: o.stake
                    for o in app.state.chain_client.get_operator_stakes()
                }
            except ChainUnreachableError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        sampler = StakeWeightedSampler(
            sample_rate=config.sample_rate, operator_stake=operator_stake,
        )
        commit = CommitReveal(
            epoch_seed=req.epoch_seed,
            validator_id=config.validator_id,
            salt=_get_or_create_salt(req.epoch_seed),
        )
        sampled = sampler.sample(accepted, commit)
        verdicts: list[dict[str, Any]] = []
        worker_url = config.worker_replay_url or None
        submitted_to_chain = 0
        queued_locally = 0
        for r in sampled:
            res = await _replay_receipt_async(
                r,
                worker_url=worker_url,
                replay_input=req.replay_inputs.get(r.job_id),
            )
            verdicts.append({
                "job_id": r.job_id,
                "operator_id": r.operator_id,
                "verdict": res.verdict.value,
                "fault": res.fault.value if res.fault else None,
            })
            if res.verdict == ReplayVerdict.MISMATCH and res.fault is not None:
                ev = SlashingEvidence(
                    operator_id=r.operator_id,
                    fault_code=res.fault,
                    evidence_hash=sha256_hex(res.detail.encode() + r.job_id.encode()),
                    related_job_id=r.job_id,
                    related_receipt_hash=r.content_hash(),
                )
                app.state.slashing_submitted.append(ev)
                queued_locally += 1
                # Primary path: submit to chain via RFC-0005 extrinsic.
                # C-04: distinguish chain-confirmed submission from
                # local-queue-only retention.
                try:
                    tx_hash = app.state.chain_client.submit_slashing_evidence(ev)
                except ChainUnreachableError as exc:
                    logger.warning("chain submission failed: %s", exc)
                    tx_hash = None
                if tx_hash is not None:
                    submitted_to_chain += 1
                # Optional secondary path: HTTP sink (used by chaos harness
                # to capture evidence even when the chain is muted). M-02:
                # surface status codes via /healthz; don't silently swallow.
                if config.slashing_endpoint:
                    try:
                        async with httpx.AsyncClient(timeout=2.0) as client:
                            sink_resp = await client.post(
                                config.slashing_endpoint,
                                json=ev.model_dump(mode="json"),
                            )
                        bucket = str(sink_resp.status_code)
                        sink_status_counts[bucket] = (
                            sink_status_counts.get(bucket, 0) + 1
                        )
                    except httpx.HTTPError as exc:
                        sink_status_counts["error"] = (
                            sink_status_counts.get("error", 0) + 1
                        )
                        logger.warning("slashing sink failed: %s", exc)
        return {
            "sampled_count": len(sampled),
            "verdicts": verdicts,
            # Total kept in memory across all epochs.
            "slashings_submitted": len(app.state.slashing_submitted),
            # C-04: per-epoch breakdown so callers can tell what actually
            # landed on chain vs. what is just queued locally.
            "submitted_to_chain": submitted_to_chain,
            "queued_locally": queued_locally,
            # N-W-01: receipt-sig rejection counters.
            "rejected_unknown_operator": unknown,
            "rejected_invalid_signature": invalid,
        }

    @app.post("/submit_weights", dependencies=[Depends(require_validator_auth)])
    async def submit_weights(req: SubmitWeightsRequest) -> dict[str, Any]:
        """Compute + sign + submit `YumaConsensus.submit_weights(epoch, vector)`.

        Reward-crank #3: this is the validator's per-epoch quality vote that
        the reward crank turns on. The validator's sr25519 key
        (`config.validator_private_key_hex`) signs the extrinsic; the live
        runtime gates it by `EpochPermittedValidators[epoch]`.

        C-04: on chain-RPC failure this RAISES (503) unless stub fallback is
        opted into, in which case the vector is computed + returned but not
        submitted (`tx_hash` is null).
        """
        if not config.validator_private_key_hex:
            raise HTTPException(
                status_code=400,
                detail=(
                    "validator_private_key_hex is not configured; the validator "
                    "cannot sign a submit_weights extrinsic without its key."
                ),
            )
        # Pull the active operator set + stakes from the chain unless the
        # caller passed an explicit override (e.g. a dry-run with a fixture).
        if req.operator_stake:
            operators = [
                OperatorStake(
                    operator_id=op_id, stake=stake, active=True,
                )
                for op_id, stake in req.operator_stake.items()
            ]
        else:
            try:
                operators = app.state.chain_client.get_operator_stakes()
            except ChainUnreachableError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
        vec = compute_weight_vector(
            epoch=req.epoch,
            operators=operators,
            faulty_operators=req.faulty_operators,
            equal_weight=req.equal_weight,
        )
        body_hex = encode_submit_weights_call(req.epoch, vec).hex()
        try:
            tx_hash = submit_yuma_weights(
                app.state.chain_client,
                epoch=req.epoch,
                vector=vec,
                validator_private_key_hex=config.validator_private_key_hex,
            )
        except ChainUnreachableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "epoch": req.epoch,
            "operator_count": len(vec.entries),
            "call_body_hex": body_hex,
            "tx_hash": tx_hash,
            "submitted_to_chain": tx_hash is not None,
        }

    @app.get("/internal/slashings", dependencies=[Depends(require_validator_auth)])
    async def slashings() -> dict[str, Any]:
        # Filter only the slashings from the most recent run.
        items = [s.model_dump(mode="json") for s in app.state.slashing_submitted]
        # convert FaultCode enum
        for it in items:
            fc = it.get("fault_code")
            if hasattr(fc, "value"):
                it["fault_code"] = fc.value
        return {"submitted": items}

    return app
