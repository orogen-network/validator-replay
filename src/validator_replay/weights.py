"""Compute + sign + submit Yuma consensus weight vectors.

This closes reward-crank item #3: `validator-replay` previously stored a
`validator_private_key_hex` it never used. It sampled receipts, replayed
them, and submitted *slashing* evidence, but never produced the per-epoch
**weight vector** that `pallet-yuma-consensus::submit_weights` consumes —
the input the reward crank actually turns on. So the validator contributed
nothing to incentive computation; the manual `drive-epoch0.cjs` driver had
to stand in with sudo-routed equal weights.

The Yuma runtime (`pallet-suite/pallets/yuma-consensus/src/lib.rs`):

    pub fn submit_weights(origin, epoch: u64, vector: Vec<(AccountId, u16)>)
        // ensure_signed; validator must be in EpochPermittedValidators[epoch]
        // and the epoch must not already be computed.

Each validator submits one `Vec<(AccountId, u16)>` per epoch — its per-operator
*quality score* in `0..=65535`. `compute_epoch_incentives` then takes a
stake-weighted clipped-median across all validators' vectors to derive each
operator's incentive share. A weight of 0 means "this operator produced
faulty work this epoch"; a weight near 65535 means "clean, full credit".

We compute the vector from the replay verdicts already produced in the
epoch run: operators with at least one MISMATCH verdict get weight 0; clean
operators get a stake-proportional score capped at `u16::MAX` (bootstrap
default: equal max weight when no stake signal, matching `drive-epoch0.cjs`).

The vector is signed with the validator's sr25519 key and submitted as a
real extrinsic via `substrate-interface` (reusing the chain client's
`SubstrateInterface`). On the live Forge chain `submit_weights` is a plain
signed call — `ensure_signed`, gated by `EpochPermittedValidators` — so no
sudo is needed for a permitted validator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from mining_types import Receipt

from validator_replay.chain import (
    ChainRpcClient,
    ChainUnreachableError,
    OperatorStake,
    _to_32,
)

logger = logging.getLogger(__name__)

# pallet-yuma-consensus is pallet index 13 in the Orogen runtime
# construct_runtime!; submit_weights is call_index(0). See
# pallet-suite/runtime/src/lib.rs.
YUMA_PALLET_INDEX = 13
YUMA_SUBMIT_WEIGHTS_CALL_INDEX = 0

U16_MAX = 0xFFFF


@dataclass(slots=True)
class WeightVector:
    """A per-operator quality score vector ready for `submit_weights`."""

    epoch: int
    # (32-byte AccountId, u16 weight) pairs, matching the runtime ABI.
    entries: list[tuple[bytes, int]]

    def as_pairs(self) -> list[tuple[bytes, int]]:
        return self.entries


def compute_weight_vector(
    epoch: int,
    operators: list[OperatorStake],
    *,
    faulty_operators: set[str],
    equal_weight: bool = True,
) -> WeightVector:
    """Build a `submit_weights` vector from the epoch's operator set + verdicts.

    - `operators` — the active operator set + stakes (from
      `ChainRpcClient.get_operator_stakes`).
    - `faulty_operators` — operator_ids (hex AccountId strings) that produced
      at least one MISMATCH verdict this epoch. These get weight 0 so the
      validator votes them no credit.
    - `equal_weight` — when True (the bootstrap default, matching
      `drive-epoch0.cjs`), every clean operator gets `u16::MAX`. When False,
      clean operators get a stake-proportional score: `stake / max_stake *
      u16::MAX`, floored at 1 so a tiny-but-clean operator isn't zeroed out.

    Only active operators are included. The vector is de-duplicated by
    operator_id; the first occurrence wins (stakes come from a storage map
    so duplicates shouldn't occur, but we defend against it).
    """
    if not operators:
        return WeightVector(epoch=epoch, entries=[])

    clean = [op for op in operators if op.active and op.operator_id not in faulty_operators]
    if not clean:
        return WeightVector(epoch=epoch, entries=[])

    if equal_weight:
        scored = [(op, U16_MAX) for op in clean]
    else:
        max_stake = max((op.stake for op in clean), default=0) or 1
        scored = [
            (op, max(1, int(op.stake * U16_MAX / max_stake)))
            for op in clean
        ]

    seen: set[str] = set()
    entries: list[tuple[bytes, int]] = []
    for op, w in scored:
        if op.operator_id in seen:
            continue
        seen.add(op.operator_id)
        entries.append((_to_32(op.operator_id), max(0, min(U16_MAX, w))))
    return WeightVector(epoch=epoch, entries=entries)


def faulty_operators_from_verdicts(
    receipts: list[Receipt],
    verdicts: list[dict[str, object]],
) -> set[str]:
    """Extract the set of operator_ids that produced at least one mismatch.

    `verdicts` is the list of dicts the `/run_epoch` endpoint returns
    (`{job_id, operator_id, verdict, fault}`). Any verdict whose `verdict`
    field is `mismatch` (the value of `ReplayVerdict.MISMATCH`) marks that
    operator faulty for the epoch.
    """
    faulty: set[str] = set()
    # receipts carries the authoritative operator_id per job_id; verdicts
    # also carry it, but we key off whichever is present.
    by_job = {r.job_id: r.operator_id for r in receipts}
    for v in verdicts:
        if str(v.get("verdict", "")).lower() != "mismatch":
            continue
        op = v.get("operator_id") or by_job.get(v.get("job_id", ""))
        if op:
            faulty.add(str(op))
    return faulty


def encode_submit_weights_call(epoch: int, vector: WeightVector) -> bytes:
    """Encode the unsigned-call body of `YumaConsensus.submit_weights`.

    Layout: pallet-index || call-index || SCALE(epoch: u64) ||
    SCALE(Vec<(AccountId32, u16)>). This is the call body only; signing
    (wrap in a signed extrinsic) is done by `substrate-interface` so the
    mortality / nonce / genesis hash are handled correctly against the live
    chain.
    """
    out = bytearray()
    out.append(YUMA_PALLET_INDEX)
    out.append(YUMA_SUBMIT_WEIGHTS_CALL_INDEX)
    # epoch: u64 little-endian
    out.extend(int.to_bytes(epoch, 8, "little"))
    # Vec<(AccountId32, u16)>: compact length || (32 bytes || u16 LE) per entry
    out.extend(_encode_compact(len(vector.entries)))
    for account, weight in vector.entries:
        if len(account) != 32:
            raise ValueError(
                f"weight-vector account must be 32 bytes, got {len(account)}"
            )
        out.extend(account)
        out.extend(int.to_bytes(max(0, min(U16_MAX, weight)), 2, "little"))
    return bytes(out)


def _encode_compact(value: int) -> bytes:
    """SCALE compact-integer encode (mirrors chain._encode_compact)."""
    if value < 0:
        raise ValueError("SCALE compact must be non-negative")
    if value < 1 << 6:
        return bytes([value << 2])
    if value < 1 << 14:
        return int.to_bytes((value << 2) | 0b01, 2, "little")
    if value < 1 << 30:
        return int.to_bytes((value << 2) | 0b10, 4, "little")
    n_bytes = ((value.bit_length() + 7) // 8)
    out = bytearray([((n_bytes - 4) << 2) | 0b11])
    out.extend(int.to_bytes(value, n_bytes, "little"))
    return bytes(out)


def submit_weights(
    client: ChainRpcClient,
    *,
    epoch: int,
    vector: WeightVector,
    validator_private_key_hex: str,
) -> str | None:
    """Sign + submit `YumaConsensus.submit_weights(epoch, vector)`.

    Returns the tx-hash hex on success. Raises `ChainUnreachableError` on
    RPC failure unless `VALIDATOR_ALLOW_STUB_CHAIN=1`, in which case
    returns `None` (the vector is logged but not submitted).

    The validator key is sr25519 (crypto_type=1). `substrate-interface`
    wraps the call in a properly signed extrinsic with the right nonce,
    mortality, and genesis hash for the live chain.
    """
    if not validator_private_key_hex:
        raise ValueError(
            "validator_private_key_hex is required to submit weights; "
            "the validator cannot participate in Yuma consensus without it.",
        )
    if not client._probe_with_retry():
        if _stub_enabled():
            logger.warning(
                "chain-node not reachable at %s — weight vector kept locally",
                client.rpc_url,
            )
            return None
        raise ChainUnreachableError(
            f"cannot submit weights: chain-node at {client.rpc_url} "
            f"unreachable after retries.",
        )
    try:
        substrate = client._substrate_client()
        from substrateinterface import Keypair  # type: ignore

        keypair = Keypair.create_from_private_key(
            validator_private_key_hex, ss58_format=42, crypto_type=1,
        )
        # substrate-interface composes the call from pallet + call name,
        # so it handles the SCALE encoding itself. We pass the decoded
        # payload: epoch (u64) + vector (list of [ss58/hex, u16]).
        call = substrate.compose_call(
            call_module="YumaConsensus",
            call_function="submit_weights",
            call_params={
                "epoch": epoch,
                "vector": [
                    [_account_to_ss32(account, substrate), int(weight)]
                    for account, weight in vector.entries
                ],
            },
        )
        extrinsic = substrate.create_signed_extrinsic(call=call, keypair=keypair)
        receipt = substrate.submit_extrinsic(extrinsic, wait_for_inclusion=True)
        if not receipt.is_ok:
            logger.warning(
                "submit_weights extrinsic failed: %s", getattr(receipt, "error_message", None)
            )
            if _stub_enabled():
                return None
            raise ChainUnreachableError(
                f"submit_weights extrinsic rejected: {getattr(receipt, 'error_message', receipt)!r}",
            )
        return getattr(receipt, "extrinsic_hash", None) or "0x"
    except ChainUnreachableError:
        raise
    except Exception as exc:
        if _stub_enabled():
            logger.warning("submit_weights failed (%s) — kept locally", exc)
            return None
        raise ChainUnreachableError(f"submit_weights failed: {exc!r}") from exc


def _account_to_ss32(account_bytes: bytes, substrate: object) -> str:
    """Convert a 32-byte AccountId to the form substrate-interface expects for
    a call param. `compose_call` accepts a hex `0x…` string for AccountId32."""
    if not isinstance(account_bytes, (bytes, bytearray)) or len(account_bytes) != 32:
        raise ValueError("account must be 32 bytes")
    return "0x" + bytes(account_bytes).hex()


def _stub_enabled() -> bool:
    import os
    return os.environ.get("VALIDATOR_ALLOW_STUB_CHAIN") == "1"
