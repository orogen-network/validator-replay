"""Replay a receipt against an inference engine and verify response hashes.

Two code paths live here:

1. **Worker-based replay** — when `worker_replay_url` is configured, the
   validator POSTs to the validator's *own* worker pool's
   `/v1/replay` endpoint and compares the returned response_hash to the
   one declared on the receipt under audit. This is the production path.
2. **Override / declarative replay** — for unit and e2e tests, callers pass
   `expected_response_hash` directly. Used by chaos scenarios to force
   matches/mismatches.

For production validators must run their own independent worker pool — see
README.md ("Worker independence"). Using the audited operator's own worker
would defeat the entire scheme.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
from mining_types import FaultCode, Receipt
from mining_types.crypto import sha256_hex


class ReplayVerdict(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    SKIPPED = "skipped"


class ReplayMode(str, Enum):
    """How a replay result is compared against the receipt under audit.

    - `EXACT` (default): byte-identical `sha256(response_text)` comparison.
      Correct only when inference is deterministic (e.g. the mock engine).
    - `TOLERANT`: compare token-id overlap and top-logprob agreement against a
      threshold. A single sub-threshold divergence is a MATCH; only divergence
      beyond `tolerance` is a MISMATCH. This is what real, non-bit-reproducible
      GPU/CPU inference requires so an honest operator is not slashed for
      ordinary floating-point / kernel non-determinism.
    """

    EXACT = "exact"
    TOLERANT = "tolerant"


# Default tolerance: claimed vs recomputed must agree on >= 98% of the signal.
DEFAULT_TOLERANCE = 0.98


def resolved_replay_mode(mode: ReplayMode | str | None = None) -> ReplayMode:
    """Resolve the replay mode from an explicit arg or `VALIDATOR_REPLAY_MODE`
    (default `exact`)."""
    if isinstance(mode, ReplayMode):
        return mode
    raw = (mode or os.environ.get("VALIDATOR_REPLAY_MODE", "") or "exact").strip().lower()
    if raw in {"", "exact"}:
        return ReplayMode.EXACT
    if raw == "tolerant":
        return ReplayMode.TOLERANT
    raise ValueError(f"unknown VALIDATOR_REPLAY_MODE {raw!r}; expected 'exact' or 'tolerant'")


def resolved_tolerance(tolerance: float | None = None) -> float:
    """Resolve the agreement threshold from an explicit arg or
    `VALIDATOR_REPLAY_TOLERANCE` (default 0.98). Clamped to [0, 1]."""
    if tolerance is None:
        raw = os.environ.get("VALIDATOR_REPLAY_TOLERANCE", "").strip()
        tolerance = float(raw) if raw else DEFAULT_TOLERANCE
    return max(0.0, min(1.0, tolerance))


def token_overlap_ratio(claimed: list[int], recomputed: list[int]) -> float:
    """Positional token-id agreement over the shorter sequence.

    Returns 1.0 for two empty sequences (nothing to disagree about). A length
    mismatch is penalised: positions only present in one sequence count as
    disagreements.
    """
    if not claimed and not recomputed:
        return 1.0
    longest = max(len(claimed), len(recomputed))
    if longest == 0:
        return 1.0
    agree = sum(1 for a, b in zip(claimed, recomputed, strict=False) if a == b)
    return agree / longest


def logprob_agreement(
    claimed: list[float], recomputed: list[float], *, abs_tol: float = 0.5
) -> float:
    """Fraction of aligned positions whose top log-probs agree within `abs_tol`.

    Real engines produce slightly different log-probs run to run, so this is a
    soft signal, not an equality check. Empty inputs agree trivially.
    """
    if not claimed and not recomputed:
        return 1.0
    longest = max(len(claimed), len(recomputed))
    if longest == 0:
        return 1.0
    agree = sum(
        1 for a, b in zip(claimed, recomputed, strict=False) if abs(a - b) <= abs_tol
    )
    return agree / longest


@dataclass(slots=True)
class TokenEvidence:
    """The signal a tolerant comparison needs, decoupled from the (pinned)
    Receipt model so this works regardless of the mining-types version in use."""

    token_ids: list[int]
    top_logprobs: list[float]

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> TokenEvidence | None:
        if not data:
            return None
        ids = data.get("token_ids")
        lps = data.get("top_logprobs")
        if ids is None and lps is None:
            return None
        return cls(
            token_ids=[int(x) for x in (ids or [])],
            top_logprobs=[float(x) for x in (lps or [])],
        )


def tolerant_score(claimed: TokenEvidence, recomputed: TokenEvidence) -> float:
    """Combined agreement score in [0, 1]: the worse of token-overlap and
    top-logprob agreement, so either signal diverging pulls the score down."""
    overlap = token_overlap_ratio(claimed.token_ids, recomputed.token_ids)
    lp = logprob_agreement(claimed.top_logprobs, recomputed.top_logprobs)
    return min(overlap, lp)


@dataclass(slots=True)
class ReplayResult:
    receipt: Receipt
    verdict: ReplayVerdict
    fault: FaultCode | None = None
    detail: str = ""
    # Tolerant-mode score (None in exact mode or when no evidence available).
    score: float | None = None


def replay_via_worker_full(
    receipt: Receipt,
    worker_url: str,
    replay_input: dict[str, Any],
) -> dict[str, Any]:
    """Issue a `POST {worker_url}/v1/replay` and return the worker's full
    response body.

    Request shape includes the original replay input, not just hashes. A
    validator cannot independently recompute the response hash from
    `request_hash` alone.

    Response shape (additive fields are optional):
        {"response_hash": "deadbeef...",
         "token_ids": [...], "top_logprobs": [...], "token_ids_digest": "..."}

    Raises `httpx.HTTPError` on transport failure — callers decide whether
    that constitutes a SKIPPED verdict or hard failure.
    """
    payload = {
        **replay_input,
        "job_id": receipt.job_id,
        "model_id": receipt.model_id,
        "customer_nonce": receipt.customer_nonce,
        "request_hash": receipt.request_hash,
    }
    token = (
        os.environ.get("VALIDATOR_WORKER_API_TOKEN", "")
        or os.environ.get("WORKER_API_TOKEN", "")
        or os.environ.get("INTERNAL_AUTH_TOKEN", "")
    ).strip()
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(
            f"{worker_url.rstrip('/')}/v1/replay",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        body = resp.json()
    return dict(body)


def replay_via_worker(
    receipt: Receipt,
    worker_url: str,
    replay_input: dict[str, Any],
) -> str:
    """Backward-compatible wrapper returning only the worker's response_hash."""
    return str(replay_via_worker_full(receipt, worker_url, replay_input)["response_hash"])


def _placeholder_replay_hash(receipt: Receipt) -> str:
    """Deterministic placeholder used when no worker is configured.

    NOT a security primitive — it just gives the in-memory tests a stable
    "expected" value to compare against.
    """
    return sha256_hex((receipt.request_hash + "::placeholder").encode())


def allow_stub_replay() -> bool:
    return os.environ.get("VALIDATOR_ALLOW_STUB_REPLAY", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _tolerant_verdict(
    receipt: Receipt,
    claimed: TokenEvidence,
    recomputed: TokenEvidence,
    tolerance: float,
) -> ReplayResult:
    """Score claimed vs recomputed evidence; MISMATCH only above tolerance.

    A single sub-threshold divergence is a MATCH (real inference drifts); only
    a divergence beyond `tolerance` is treated as material and slashable. The
    daemon escalates *repeated* material mismatches for the same operator (see
    `MaterialMismatchTracker`).
    """
    score = tolerant_score(claimed, recomputed)
    if score < tolerance:
        return ReplayResult(
            receipt=receipt,
            verdict=ReplayVerdict.MISMATCH,
            fault=FaultCode.WRONG_RESPONSE,
            detail=f"tolerant divergence: score {score:.4f} < tolerance {tolerance:.4f}",
            score=score,
        )
    return ReplayResult(
        receipt=receipt,
        verdict=ReplayVerdict.MATCH,
        detail=f"tolerant match: score {score:.4f} >= tolerance {tolerance:.4f}",
        score=score,
    )


def replay_receipt(
    receipt: Receipt,
    *,
    expected_response_hash: str | None = None,
    expected_model_weight_hash: str | None = None,
    worker_url: str | None = None,
    replay_input: dict[str, Any] | None = None,
    mode: ReplayMode | str | None = None,
    tolerance: float | None = None,
    claimed_token_evidence: dict[str, Any] | None = None,
    recomputed_token_evidence: dict[str, Any] | None = None,
) -> ReplayResult:
    """Compare declared receipt vs. replay output.

    Comparison mode (`mode`, else env `VALIDATOR_REPLAY_MODE`, default
    `exact`):

    - `exact`: byte-identical `response_hash` comparison (current behaviour).
    - `tolerant`: compare token-overlap / top-logprob agreement against
      `tolerance` (else `VALIDATOR_REPLAY_TOLERANCE`, default 0.98). Requires
      both a *claimed* and a *recomputed* TokenEvidence; if either is missing
      it falls back to the exact comparison so the result is never weaker than
      exact mode by accident.

    Decision order for the comparison source:
      1. `expected_response_hash` (+ optional `recomputed_token_evidence`) — test override.
      2. `worker_url` + `replay_input` — live replay against the validator's worker.
      3. explicit `VALIDATOR_ALLOW_STUB_REPLAY=1` dev/test mode.
      4. otherwise fail closed with a skipped verdict; never self-attest.
    """
    if expected_model_weight_hash and expected_model_weight_hash != receipt.model_weight_hash:
        return ReplayResult(
            receipt=receipt,
            verdict=ReplayVerdict.MISMATCH,
            fault=FaultCode.WRONG_MODEL,
            detail="model_weight_hash mismatch",
        )

    resolved_mode = resolved_replay_mode(mode)
    # The claimed evidence may be supplied directly or carried in the input.
    claimed_raw = claimed_token_evidence
    if claimed_raw is None and replay_input is not None:
        claimed_raw = replay_input.get("claimed_token_evidence")
    recomputed_raw = recomputed_token_evidence

    if expected_response_hash is not None:
        replay_response = expected_response_hash
    elif worker_url:
        if replay_input is None:
            return ReplayResult(
                receipt=receipt,
                verdict=ReplayVerdict.SKIPPED,
                fault=None,
                detail="replay input required for worker replay",
            )
        # In tolerant mode we need the worker's full body (token ids / logprobs)
        # for the recomputed side; in exact mode the response_hash suffices.
        # `replay_via_worker` stays the call site so existing tests that patch
        # it keep intercepting the exact path.
        want_full = resolved_mode is ReplayMode.TOLERANT and recomputed_raw is None
        try:
            if want_full:
                body = replay_via_worker_full(receipt, worker_url, replay_input)
                replay_response = str(body["response_hash"])
                recomputed_raw = body
            else:
                replay_response = replay_via_worker(receipt, worker_url, replay_input)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in {400, 422}:
                return ReplayResult(
                    receipt=receipt,
                    verdict=ReplayVerdict.MISMATCH,
                    fault=FaultCode.WRONG_RESPONSE,
                    detail=f"worker rejected replay input: HTTP {exc.response.status_code}",
                )
            return ReplayResult(
                receipt=receipt,
                verdict=ReplayVerdict.SKIPPED,
                fault=None,
                detail=f"worker rejected replay request: HTTP {exc.response.status_code}",
            )
        except httpx.HTTPError as exc:
            return ReplayResult(
                receipt=receipt,
                verdict=ReplayVerdict.SKIPPED,
                fault=None,
                detail=f"worker unreachable: {exc}",
            )
    elif allow_stub_replay():
        replay_response = receipt.response_hash
    else:
        return ReplayResult(
            receipt=receipt,
            verdict=ReplayVerdict.SKIPPED,
            fault=None,
            detail="worker_replay_url required unless VALIDATOR_ALLOW_STUB_REPLAY=1",
        )

    if resolved_mode is ReplayMode.TOLERANT:
        claimed_ev = TokenEvidence.from_mapping(claimed_raw)
        recomputed_ev = TokenEvidence.from_mapping(recomputed_raw)
        if claimed_ev is not None and recomputed_ev is not None:
            return _tolerant_verdict(
                receipt, claimed_ev, recomputed_ev, resolved_tolerance(tolerance)
            )
        # Insufficient evidence for a tolerant compare: fall back to exact so
        # we never silently weaken the check.

    if replay_response != receipt.response_hash:
        return ReplayResult(
            receipt=receipt,
            verdict=ReplayVerdict.MISMATCH,
            fault=FaultCode.WRONG_RESPONSE,
            detail="response_hash mismatch",
        )
    return ReplayResult(receipt=receipt, verdict=ReplayVerdict.MATCH)


@dataclass
class MaterialMismatchTracker:
    """Track repeated material (tolerant) mismatches per operator.

    A single sub-threshold divergence under tolerant mode is treated as a
    MATCH by `replay_receipt`. When the daemon does see a tolerant MISMATCH it
    records it here; an operator is only considered *materially* faulty once it
    accumulates `threshold` mismatches, which guards against slashing on a
    one-off divergence that slipped just under the tolerance line.
    """

    threshold: int = 3
    _counts: dict[str, int] = field(default_factory=dict)

    def record(self, operator_id: str) -> int:
        self._counts[operator_id] = self._counts.get(operator_id, 0) + 1
        return self._counts[operator_id]

    def is_material(self, operator_id: str) -> bool:
        return self._counts.get(operator_id, 0) >= self.threshold

    def reset(self, operator_id: str) -> None:
        self._counts.pop(operator_id, None)


__all__ = [
    "DEFAULT_TOLERANCE",
    "MaterialMismatchTracker",
    "ReplayMode",
    "ReplayResult",
    "ReplayVerdict",
    "TokenEvidence",
    "_placeholder_replay_hash",
    "allow_stub_replay",
    "logprob_agreement",
    "replay_receipt",
    "replay_via_worker",
    "replay_via_worker_full",
    "resolved_replay_mode",
    "resolved_tolerance",
    "token_overlap_ratio",
    "tolerant_score",
]
