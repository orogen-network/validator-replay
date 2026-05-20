"""Validator configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_CHAIN_RPC_URL = "ws://127.0.0.1:9944"
DEFAULT_REPLAY_INPUT_MAX_BYTES = 1_048_576
REPLAY_INPUT_POLICIES = {"ephemeral"}


def _default_chain_rpc_url() -> str:
    """Resolve the chain RPC URL from environment, falling back to the
    canonical local dev URL. Surfaced as a module-level default so tests can
    monkey-patch the env without touching the dataclass directly."""
    return os.environ.get("VALIDATOR_CHAIN_RPC_URL", DEFAULT_CHAIN_RPC_URL)


@dataclass(slots=True)
class ValidatorConfig:
    """All runtime knobs for the validator-replay daemon.

    - `chain_rpc_url`  — WS/HTTP endpoint of the chain-node. Defaults to
      `VALIDATOR_CHAIN_RPC_URL` env (or `ws://127.0.0.1:9944`).
    - `worker_replay_url` — base URL of the validator's *own* worker pool
      (independent of the operator under audit) that we ask to re-issue the
      same inference for cross-comparison.
    - `slashing_endpoint` — fallback HTTP endpoint that accepts slashing
      evidence as JSON when `chain_rpc_url` is unreachable. Useful in tests
      and for offline analysis.
    - `replay_input_policy` — privacy policy for original replay inputs. The
      only implemented production-safe policy is `ephemeral`: inputs are
      accepted in the request body, capped, forwarded to the validator-owned
      replay worker, and not retained or returned by this service.
    """

    validator_id: str
    validator_private_key_hex: str
    sample_rate: float = 0.25
    epoch_seed: str = ""  # filled per-epoch by chain RPC
    chain_rpc_url: str = ""  # default resolved at runtime; see `resolved_chain_rpc_url()`
    worker_replay_url: str = ""  # base URL of validator's own worker pool
    slashing_endpoint: str = ""  # HTTP fallback for slashing evidence
    replay_input_policy: str = ""
    replay_input_max_bytes: int = DEFAULT_REPLAY_INPUT_MAX_BYTES
    # L-04: cache the resolved URL so a hostile env mutation between calls
    # cannot switch the validator's target chain mid-epoch.
    _resolved_url_cache: str = ""

    def resolved_chain_rpc_url(self) -> str:
        if self._resolved_url_cache:
            return self._resolved_url_cache
        resolved = self.chain_rpc_url or _default_chain_rpc_url()
        # slots=True means we can still mutate, but the field exists.
        object.__setattr__(self, "_resolved_url_cache", resolved)
        return resolved

    def resolved_replay_input_policy(self) -> str:
        return (
            self.replay_input_policy
            or os.environ.get("VALIDATOR_REPLAY_INPUT_POLICY", "")
            or "ephemeral"
        ).strip().lower()

    def validate_replay_input_policy(self, *, production: bool) -> None:
        policy = self.resolved_replay_input_policy()
        if policy not in REPLAY_INPUT_POLICIES:
            allowed = ", ".join(sorted(REPLAY_INPUT_POLICIES))
            raise RuntimeError(
                f"unsupported replay input policy {policy!r}; allowed: {allowed}"
            )
        if self.replay_input_max_bytes <= 0:
            raise RuntimeError("replay_input_max_bytes must be positive")
        if production and self.worker_replay_url and not (
            self.replay_input_policy or os.environ.get("VALIDATOR_REPLAY_INPUT_POLICY", "")
        ):
            raise RuntimeError(
                "production worker replay requires VALIDATOR_REPLAY_INPUT_POLICY=ephemeral"
            )
