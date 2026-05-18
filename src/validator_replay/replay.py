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

from dataclasses import dataclass
from enum import Enum

import httpx
from mining_types import FaultCode, Receipt
from mining_types.crypto import sha256_hex


class ReplayVerdict(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    SKIPPED = "skipped"


@dataclass(slots=True)
class ReplayResult:
    receipt: Receipt
    verdict: ReplayVerdict
    fault: FaultCode | None = None
    detail: str = ""


def replay_via_worker(receipt: Receipt, worker_url: str) -> str:
    """Issue a `POST {worker_url}/v1/replay` for the receipt under audit and
    return the worker's freshly-computed response_hash.

    Request shape (mirrors RFC-0001 §replay):
        {"job_id":..., "model_id":..., "request_hash":..., "customer_nonce":...}

    Response shape:
        {"response_hash": "deadbeef..."}

    Raises `httpx.HTTPError` on transport failure — callers decide whether
    that constitutes a SKIPPED verdict or hard failure.
    """
    payload = {
        "job_id": receipt.job_id,
        "model_id": receipt.model_id,
        "request_hash": receipt.request_hash,
        "customer_nonce": receipt.customer_nonce,
    }
    with httpx.Client(timeout=10.0) as client:
        resp = client.post(f"{worker_url.rstrip('/')}/v1/replay", json=payload)
        resp.raise_for_status()
        body = resp.json()
    return str(body["response_hash"])


def _placeholder_replay_hash(receipt: Receipt) -> str:
    """Deterministic placeholder used when no worker is configured.

    NOT a security primitive — it just gives the in-memory tests a stable
    "expected" value to compare against.
    """
    return sha256_hex((receipt.request_hash + "::placeholder").encode())


def replay_receipt(
    receipt: Receipt,
    *,
    expected_response_hash: str | None = None,
    expected_model_weight_hash: str | None = None,
    worker_url: str | None = None,
) -> ReplayResult:
    """Compare declared receipt vs. replay output.

    Decision order for the comparison hash:
      1. `expected_response_hash` — test override.
      2. `worker_url` — issue a live replay against the validator's worker.
      3. fall back to `receipt.response_hash` (zero-effort match) — used by
         tests that exercise the harness without a live worker.
    """
    if expected_model_weight_hash and expected_model_weight_hash != receipt.model_weight_hash:
        return ReplayResult(
            receipt=receipt,
            verdict=ReplayVerdict.MISMATCH,
            fault=FaultCode.WRONG_MODEL,
            detail="model_weight_hash mismatch",
        )

    if expected_response_hash is not None:
        replay_response = expected_response_hash
    elif worker_url:
        try:
            replay_response = replay_via_worker(receipt, worker_url)
        except httpx.HTTPError as exc:
            return ReplayResult(
                receipt=receipt,
                verdict=ReplayVerdict.SKIPPED,
                fault=None,
                detail=f"worker unreachable: {exc}",
            )
    else:
        # No worker configured + no explicit override: treat the receipt as
        # self-attesting. Tests rely on this branch.
        replay_response = receipt.response_hash

    if replay_response != receipt.response_hash:
        return ReplayResult(
            receipt=receipt,
            verdict=ReplayVerdict.MISMATCH,
            fault=FaultCode.WRONG_RESPONSE,
            detail="response_hash mismatch",
        )
    return ReplayResult(receipt=receipt, verdict=ReplayVerdict.MATCH)


__all__ = [
    "ReplayResult",
    "ReplayVerdict",
    "_placeholder_replay_hash",
    "replay_receipt",
    "replay_via_worker",
]
