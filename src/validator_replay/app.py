"""Validator replay HTTP API.

Endpoints (deliberately RPC-style, since real chain interaction is mocked):
- `POST /sample`     — submit a list of receipts + epoch_seed → returns sampled set.
- `POST /replay`     — replay sampled receipts; returns verdicts.
- `POST /run_epoch`  — convenience: sample + replay + (mock) slashing call in one shot.
- `GET  /healthz`
"""

from __future__ import annotations

import logging
import secrets
from typing import Any

import httpx
from fastapi import FastAPI
from mining_types import Receipt, SlashingEvidence
from mining_types.crypto import sha256_hex, verify_ed25519
from pydantic import BaseModel

from validator_replay.chain import ChainRpcClient, ChainUnreachableError
from validator_replay.config import ValidatorConfig
from validator_replay.registry import OperatorRegistry, allow_unsigned_ingest
from validator_replay.replay import ReplayVerdict, replay_receipt
from validator_replay.sampler import CommitReveal, StakeWeightedSampler

logger = logging.getLogger(__name__)


class SampleRequest(BaseModel):
    receipts: list[Receipt]
    epoch_seed: str
    operator_stake: dict[str, int] = {}


class ReplayRequest(BaseModel):
    receipts: list[Receipt]
    # If provided, override response_hash to simulate mismatch
    override_response_hash_for: dict[str, str] = {}
    override_model_weight_hash_for: dict[str, str] = {}


class RunEpochRequest(BaseModel):
    receipts: list[Receipt]
    epoch_seed: str
    operator_stake: dict[str, int] = {}
    force_mismatch_operator: str | None = None


def build_app(
    config: ValidatorConfig,
    registry: OperatorRegistry | None = None,
) -> FastAPI:
    app = FastAPI(title="validator-replay", version="0.1.0")
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

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "ok": True,
            "validator_id": config.validator_id,
            "chain_rpc_url": config.resolved_chain_rpc_url(),
            "sink_status_counts": dict(sink_status_counts),
        }

    @app.get("/internal/operators")
    async def operators() -> dict[str, Any]:
        """Surface the chain-derived operator set + stakes.

        C-04: on chain-RPC failure this RAISES (returns 503) unless stub
        fallback was opted into. We no longer silently substitute stub data.
        """
        try:
            ops = app.state.chain_client.get_operator_stakes()
        except ChainUnreachableError as exc:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "rpc_url": config.resolved_chain_rpc_url(),
            "operators": [
                {"operator_id": o.operator_id, "stake": o.stake, "active": o.active}
                for o in ops
            ],
        }

    @app.post("/sample")
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

    @app.post("/replay")
    async def replay(req: ReplayRequest) -> dict[str, Any]:
        # N-W-01: refuse to act on forged receipts. Unknown / invalid
        # signatures are dropped before replay.
        accepted, unknown, invalid = _verify_receipts(req.receipts)
        results: list[dict[str, Any]] = []
        worker_url = config.worker_replay_url or None
        for r in accepted:
            override_resp = req.override_response_hash_for.get(r.operator_id)
            override_w = req.override_model_weight_hash_for.get(r.operator_id)
            res = replay_receipt(
                r,
                expected_response_hash=override_resp,
                expected_model_weight_hash=override_w,
                worker_url=worker_url if override_resp is None else None,
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

    @app.post("/run_epoch")
    async def run_epoch(req: RunEpochRequest) -> dict[str, Any]:
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
                from fastapi import HTTPException

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
            override = (
                "deadbeef" * 8
                if req.force_mismatch_operator
                and r.operator_id == req.force_mismatch_operator
                else None
            )
            res = replay_receipt(
                r,
                expected_response_hash=override,
                worker_url=worker_url if override is None else None,
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

    @app.get("/internal/slashings")
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
