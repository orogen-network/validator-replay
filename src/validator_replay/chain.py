"""Real chain RPC client for validator-replay.

Handles two concerns:

1. **Operator-stake snapshots** — pulls the active operator set + stake
   from `pallet-operator-stake::Operators` for the current epoch.

2. **Slashing-evidence submission** — builds an extrinsic body matching
   RFC-0005 (`pallet-slashing::submit_slashing_evidence`) and submits it
   to the chain.

C-04 fix: stub fallback is now OPT-IN via `VALIDATOR_ALLOW_STUB_CHAIN=1`.
On RPC failure we retry with exponential backoff and then RAISE
`ChainUnreachableError` — we DO NOT silently substitute attacker-shaped
stub data. The auto-written stub file under the package install path is
also removed (L-05).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from mining_types import SlashingEvidence

logger = logging.getLogger(__name__)


# 100 ms per project spec — short enough that a missing chain-node does not
# block the validator's hot path.
RPC_PROBE_TIMEOUT_S = 0.1
RPC_CALL_TIMEOUT_S = 5.0

# Stub data path: read-only resource bundled with the package; never written
# at import time. Override with VALIDATOR_STUB_DATA_DIR if you need to point
# at a test fixture.
STUB_DIR = Path(__file__).resolve().parent / "_stub_data"


class ChainUnreachableError(RuntimeError):
    """Raised when the chain RPC endpoint is unreachable and stub fallback
    is not explicitly opted into via `VALIDATOR_ALLOW_STUB_CHAIN=1`."""


def _stub_enabled() -> bool:
    return os.environ.get("VALIDATOR_ALLOW_STUB_CHAIN") == "1"


@dataclass(slots=True)
class OperatorStake:
    operator_id: str
    stake: int
    active: bool = True


def _stub_operators() -> list[OperatorStake]:
    """Read-only stub used when `VALIDATOR_ALLOW_STUB_CHAIN=1` is set.

    L-05: this is now a read-only resource lookup; the file is NOT
    auto-written at import time. If the stub file does not exist, returns
    an empty list (caller is expected to seed it explicitly).
    """
    path = Path(os.environ.get("VALIDATOR_STUB_DATA_DIR", str(STUB_DIR))) / "operators.json"
    if not path.exists():
        return []
    items: list[dict[str, Any]] = json.loads(path.read_text())
    return [
        OperatorStake(
            operator_id=str(it["operator_id"]),
            stake=int(it["stake"]),
            active=bool(it.get("active", True)),
        )
        for it in items
    ]


class ChainRpcClient:
    """Thin JSON-RPC client over HTTP/WS — uses HTTP for queries (subxt-rpcs
    style) and HTTP `author_submitExtrinsic` for evidence submission.

    The client is created per-call to avoid a long-lived connection that
    needs supervision; validator-replay calls these <1×/minute so the
    handshake cost is negligible.
    """

    def __init__(self, rpc_url: str) -> None:
        # M-05: refuse plaintext ws:// for non-loopback targets.
        _enforce_ws_safety(rpc_url)
        self.rpc_url = rpc_url
        self._http_url = _ws_to_http(rpc_url)
        # Retry policy (C-04): exponential backoff up to ~700 ms total.
        self._retry_backoffs_s = (0.1, 0.2, 0.4)

    # ------------------------------------------------------------------ probe

    def _probe(self) -> bool:
        """Return True if the chain-node responds within `RPC_PROBE_TIMEOUT_S`."""
        try:
            with httpx.Client(timeout=RPC_PROBE_TIMEOUT_S) as client:
                resp = client.post(
                    self._http_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "system_chain",
                        "params": [],
                    },
                )
                return resp.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def _probe_with_retry(self) -> bool:
        """Probe with exponential backoff retry (C-04)."""
        if self._probe():
            return True
        for delay in self._retry_backoffs_s:
            time.sleep(delay)
            if self._probe():
                return True
        return False

    # ---------------------------------------------------- operators
    def get_operator_stakes(self) -> list[OperatorStake]:
        """Read the active operator set + per-operator stake.

        Raises `ChainUnreachableError` on RPC failure unless
        `VALIDATOR_ALLOW_STUB_CHAIN=1` is set. The implicit fallback to
        stub operator data has been removed (C-04) — silently substituting
        attacker-shaped operator IDs into slashing extrinsics is what made
        the validator forge-targetable in the first place.
        """
        if not self._probe_with_retry():
            if _stub_enabled():
                logger.warning(
                    "chain-node not reachable at %s — stub fallback enabled, "
                    "returning stub data (VALIDATOR_ALLOW_STUB_CHAIN=1)",
                    self.rpc_url,
                )
                return _stub_operators()
            raise ChainUnreachableError(
                f"chain-node at {self.rpc_url} is unreachable after retries; "
                f"set VALIDATOR_ALLOW_STUB_CHAIN=1 to fall back to stub data "
                f"(test-only).",
            )
        try:
            with httpx.Client(timeout=RPC_CALL_TIMEOUT_S) as client:
                # We use a runtime API call exposed by pallet-operator-stake:
                # `OperatorStakeApi_operators` — returns SCALE-encoded
                # `Vec<(AccountId, Balance)>`. The chain-side runtime API
                # decode lives in pallet-suite; here we just fetch the raw
                # bytes and let SCALE-decode happen client-side.
                resp = client.post(
                    self._http_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "state_call",
                        "params": ["OperatorStakeApi_operators", "0x"],
                    },
                )
                resp.raise_for_status()
                envelope = resp.json()
                if "error" in envelope:
                    if _stub_enabled():
                        logger.warning(
                            "OperatorStakeApi_operators error: %s — using stub",
                            envelope["error"],
                        )
                        return _stub_operators()
                    raise ChainUnreachableError(
                        f"OperatorStakeApi_operators returned error: "
                        f"{envelope['error']!r}",
                    )
                return _decode_operators_hex(envelope.get("result", "0x"))
        except (httpx.HTTPError, OSError) as exc:
            if _stub_enabled():
                logger.warning("OperatorStakeApi RPC failed (%s) — using stub", exc)
                return _stub_operators()
            raise ChainUnreachableError(
                f"OperatorStakeApi RPC failed: {exc!r}",
            ) from exc

    # ---------------------------------------------------- slashing
    def submit_slashing_evidence(self, evidence: SlashingEvidence) -> str | None:
        """Submit a `pallet-slashing::submit_slashing_evidence(ev)` extrinsic.

        Returns the tx-hash hex on success. Raises `ChainUnreachableError`
        on RPC failure unless `VALIDATOR_ALLOW_STUB_CHAIN=1`, in which case
        returns `None` and the caller keeps the evidence in memory.
        """
        if not self._probe_with_retry():
            if _stub_enabled():
                logger.warning(
                    "chain-node not reachable at %s — slashing evidence kept in memory",
                    self.rpc_url,
                )
                return None
            raise ChainUnreachableError(
                f"cannot submit slashing evidence: chain-node at "
                f"{self.rpc_url} unreachable after retries.",
            )
        body = _encode_slashing_extrinsic(evidence)
        try:
            with httpx.Client(timeout=RPC_CALL_TIMEOUT_S) as client:
                resp = client.post(
                    self._http_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 1,
                        "method": "author_submitExtrinsic",
                        "params": ["0x" + body.hex()],
                    },
                )
                resp.raise_for_status()
                envelope = resp.json()
                if "error" in envelope:
                    logger.warning(
                        "author_submitExtrinsic returned error: %s",
                        envelope["error"],
                    )
                    if _stub_enabled():
                        return None
                    raise ChainUnreachableError(
                        f"author_submitExtrinsic returned error: "
                        f"{envelope['error']!r}",
                    )
                return envelope.get("result")
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("submit_slashing_evidence failed: %s", exc)
            if _stub_enabled():
                return None
            raise ChainUnreachableError(
                f"submit_slashing_evidence failed: {exc!r}",
            ) from exc


def _enforce_ws_safety(url: str) -> None:
    """M-05: refuse plaintext `ws://` (or `http://`) for non-loopback hosts."""
    if not (url.startswith("ws://") or url.startswith("http://")):
        return
    # Strip scheme + path; the host is what we check.
    rest = url.split("://", 1)[1]
    host = rest.split("/", 1)[0].split(":", 1)[0]
    if host in ("127.0.0.1", "localhost", "::1"):
        return
    if os.environ.get("VALIDATOR_ALLOW_PLAINTEXT_RPC") == "1":
        logger.warning(
            "plaintext RPC URL %r allowed via VALIDATOR_ALLOW_PLAINTEXT_RPC=1",
            url,
        )
        return
    raise ValueError(
        f"refusing plaintext chain RPC URL {url!r} to non-loopback host "
        f"{host!r}; use wss:// or set VALIDATOR_ALLOW_PLAINTEXT_RPC=1.",
    )


def _ws_to_http(url: str) -> str:
    if url.startswith("ws://"):
        return "http://" + url[len("ws://"):]
    if url.startswith("wss://"):
        return "https://" + url[len("wss://"):]
    return url


def _decode_operators_hex(hex_str: str) -> list[OperatorStake]:
    """Decode the `OperatorStakeApi_operators` response.

    The chain-side encoding is `SCALE(Vec<(AccountId32, Balance)>)`. The
    full SCALE codec is in `parity-scale-codec` (Rust); on the Python side
    we hand-decode the layout for this single API since it's stable per
    RFC-0009 and the dependency surface should stay small.
    """
    if not hex_str or hex_str == "0x":
        return []
    raw = bytes.fromhex(hex_str.removeprefix("0x"))
    n, offset = _decode_compact(raw, 0)
    out: list[OperatorStake] = []
    for _ in range(n):
        account = raw[offset : offset + 32]
        offset += 32
        balance = int.from_bytes(raw[offset : offset + 16], "little")
        offset += 16
        out.append(
            OperatorStake(
                operator_id=account.hex(),
                stake=balance,
                active=True,
            )
        )
    return out


def _decode_compact(buf: bytes, offset: int) -> tuple[int, int]:
    """Decode a SCALE compact integer. Returns (value, new_offset)."""
    first = buf[offset]
    mode = first & 0b11
    if mode == 0b00:
        return first >> 2, offset + 1
    if mode == 0b01:
        val = int.from_bytes(buf[offset : offset + 2], "little") >> 2
        return val, offset + 2
    if mode == 0b10:
        val = int.from_bytes(buf[offset : offset + 4], "little") >> 2
        return val, offset + 4
    # mode == 0b11 → big-int (length in the top 6 bits of the first byte + 4).
    n_bytes = (first >> 2) + 4
    val = int.from_bytes(buf[offset + 1 : offset + 1 + n_bytes], "little")
    return val, offset + 1 + n_bytes


def _encode_compact(value: int) -> bytes:
    """Encode a non-negative integer in SCALE compact form."""
    if value < 0:
        raise ValueError("SCALE compact must be non-negative")
    if value < 1 << 6:
        return bytes([value << 2])
    if value < 1 << 14:
        return int.to_bytes((value << 2) | 0b01, 2, "little")
    if value < 1 << 30:
        return int.to_bytes((value << 2) | 0b10, 4, "little")
    raise ValueError("compact-bigint encoding not implemented (not needed here)")


def _encode_slashing_extrinsic(ev: SlashingEvidence) -> bytes:
    """Encode the body of `pallet-slashing::submit_slashing_evidence(ev)`.

    This is the unsigned-call body (pallet-index byte || call-index byte ||
    SCALE(args)). It mirrors the RFC-0005 ABI:

        struct EvidenceCall {
            operator_id: AccountId32,        // 32 bytes
            fault_code:  u8,                 // enum variant index per RFC-0005
            evidence_hash: H256,             // 32 bytes
            related_job_id: Option<H256>,    // 1 + 32 bytes
            related_receipt_hash: Option<H256>,
        }

    Pallet/call indices are governed by the runtime `construct_runtime!`.
    The Orogen runtime assigns `pallet_slashing = 15` (see
    pallet-suite/runtime/src/lib.rs) and `submit_slashing_evidence` is
    `call_index(0)`. The previous value `pallet=42` was a placeholder that
    would have dispatched to a non-existent pallet on the live chain.
    """
    PALLET_INDEX = 15
    CALL_INDEX = 0

    out = bytearray()
    out.append(PALLET_INDEX)
    out.append(CALL_INDEX)

    # operator_id: AccountId32 — we accept either hex 0x… or the raw 32-byte
    # form. SS58 addresses are converted at the wallet layer, not here.
    op_bytes = _to_32(ev.operator_id)
    out.extend(op_bytes)

    out.append(_fault_code_index(ev.fault_code.value))

    out.extend(_to_32(ev.evidence_hash))

    _append_option_h256(out, ev.related_job_id)
    _append_option_h256(out, ev.related_receipt_hash)

    return bytes(out)


def _to_32(s: str) -> bytes:
    """Normalise to a 32-byte chain account ID.

    H-08: only 64-hex-char strings (optionally `0x`-prefixed) are accepted.
    The legacy UTF-8 zero-pad path is GONE — non-hex operator IDs collided
    on chain when their first 32 UTF-8 bytes matched, and attacker-shaped
    fixture names (e.g. `stub-op-1`) were silently encoded into evidence
    extrinsics.

    Raises ValueError on any non-hex or wrong-length input.
    """
    cleaned = s.removeprefix("0x")
    if len(cleaned) != 64:
        raise ValueError(
            f"operator_id must be 32-byte hex (64 chars), got {len(cleaned)} chars",
        )
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"operator_id is not valid hex: {s!r}",
        ) from exc


def _append_option_h256(out: bytearray, value: str | None) -> None:
    if value is None:
        out.append(0)  # Option::None
    else:
        out.append(1)  # Option::Some
        out.extend(_to_32(value))


# RFC-0005 fault-code variant ordering. Must match the on-chain enum.
_FAULT_CODES = [
    "WrongModel",
    "WrongResponse",
    "LogProbDrift",
    "CacheReplay",
    "QuantizationSwap",
    "KernelPackMismatch",
    "DeviceCertCollision",
    "HeartbeatMiss",
    "AttestationStale",
    "SanctionsHit",
    "ValidatorCollusion",
    "FakeBurn",
    "BatchOvercommit",
]


def _fault_code_index(code: str) -> int:
    try:
        return _FAULT_CODES.index(code)
    except ValueError as exc:
        raise ValueError(f"unknown FaultCode {code!r}") from exc


# L-05: the previous `_ensure_stub()` auto-write at import time has been
# removed. Writing to the package install path on import can either fail
# (read-only mount, immutable container layer) or, in a writable install,
# be replaced by an attacker between imports. Tests that need stub data
# now create it explicitly via the `VALIDATOR_STUB_DATA_DIR` env or by
# writing the file under tmp_path.


# Module-level "now" helper (mockable from tests).
def _now_ms() -> int:
    return int(time.time() * 1000)
