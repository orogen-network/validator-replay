"""Operator public-key registry for signature verification.

N-W-01: every downstream ingest endpoint must verify the ed25519 signature
on incoming payloads against a known public key. This module is the local
trust anchor for those checks.

In production this is sourced from `pallet-operator-registry` via a chain
client. For the skeleton + tests, two loaders are provided:

  - in-process `register(...)`
  - JSON file (or inline JSON) via the `OPERATOR_REGISTRY_JSON` env var
    (mapping `{operator_id: public_key_hex}`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class OperatorRegistry:
    def __init__(self, operators: dict[str, str] | None = None) -> None:
        self._operators: dict[str, str] = dict(operators or {})

    @classmethod
    def from_env(cls) -> OperatorRegistry:
        return cls(operators=_load_json_env("OPERATOR_REGISTRY_JSON"))

    def register(self, operator_id: str, public_key_hex: str) -> None:
        self._operators[operator_id] = public_key_hex

    def get(self, operator_id: str) -> str | None:
        return self._operators.get(operator_id)

    def __contains__(self, operator_id: str) -> bool:
        return operator_id in self._operators

    def __len__(self) -> int:
        return len(self._operators)


def _load_json_env(var: str) -> dict[str, str]:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        data = json.loads(raw)
    else:
        p = Path(raw)
        if not p.exists():
            return {}
        data = json.loads(p.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{var} must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def allow_unsigned_ingest() -> bool:
    """Dev/test escape hatch — production must NEVER set this."""
    if os.environ.get("OROGEN_ENV", "").strip().lower() == "production":
        return False
    return os.environ.get("ALLOW_UNSIGNED_INGEST", "").strip() == "1"
