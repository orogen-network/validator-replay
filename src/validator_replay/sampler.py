"""Stake-weighted commit-reveal sampler.

Algorithm (placeholder for RFC-0006):
- Commit phase: validator publishes BLAKE2(epoch_seed || validator_id || salt).
- Reveal phase: validator publishes (epoch_seed, salt).
- Sample selection: Fisher-Yates over (operator, stake_weight) pairs, deterministic
  PRNG keyed on BLAKE2(epoch_seed || validator_id || salt).
- Sample size = ceil(sample_rate × len(receipts)).
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from mining_types import Receipt


@dataclass(slots=True)
class CommitReveal:
    epoch_seed: str
    validator_id: str
    salt: str

    def commit(self) -> str:
        return hashlib.blake2b(
            (self.epoch_seed + self.validator_id + self.salt).encode(),
            digest_size=32,
        ).hexdigest()

    def reveal(self) -> tuple[str, str]:
        return self.epoch_seed, self.salt


def _seed_from(commit: CommitReveal) -> int:
    return int.from_bytes(
        hashlib.blake2b(
            (commit.epoch_seed + commit.validator_id + commit.salt).encode(),
            digest_size=8,
        ).digest(),
        "big",
        signed=False,
    )


class StakeWeightedSampler:
    """Fisher-Yates over stake-weighted bag, deterministic by epoch."""

    def __init__(
        self, sample_rate: float = 0.25, operator_stake: dict[str, int] | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be in [0,1]")
        self.sample_rate = sample_rate
        self.operator_stake = operator_stake or {}

    def sample(self, receipts: list[Receipt], commit: CommitReveal) -> list[Receipt]:
        if not receipts:
            return []
        rng = random.Random(_seed_from(commit))
        # Build weighted indices: heavier-stake operators get more entries.
        weights = [
            max(1, self.operator_stake.get(r.operator_id, 1)) for r in receipts
        ]
        # Fisher-Yates over (idx, weight) sampling without replacement.
        n_sample = max(1, int(len(receipts) * self.sample_rate))
        indices = list(range(len(receipts)))
        rng.shuffle(indices)
        # Bias the shuffle by weight: pick first N with weighted probability.
        chosen: list[int] = []
        bag = list(zip(indices, weights, strict=True))
        while bag and len(chosen) < n_sample:
            total = sum(w for _, w in bag)
            target = rng.uniform(0.0, total)
            acc = 0.0
            for i, (idx, w) in enumerate(bag):
                acc += w
                if acc >= target:
                    chosen.append(idx)
                    bag.pop(i)
                    break
        return [receipts[i] for i in chosen]
