"""Prefix-reuse accounting for chunked reasoning trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def common_prefix_length(previous: Sequence[int], current: Sequence[int]) -> int:
    """Return the number of unchanged leading token ids."""
    shared = 0
    for previous_token, current_token in zip(previous, current):
        if previous_token != current_token:
            break
        shared += 1
    return shared


@dataclass
class PrefixCacheReceipt:
    """Conservative receipt for prefix tokens eligible for server-side reuse."""

    turns: int = 0
    cacheable_tokens: int = 0
    submitted_tokens: int = 0

    def observe(self, previous: Sequence[int] | None, current: Sequence[int]) -> None:
        self.turns += 1
        self.submitted_tokens += len(current)
        if previous is not None:
            self.cacheable_tokens += common_prefix_length(previous, current)

    @property
    def eligible_fraction(self) -> float:
        if self.submitted_tokens == 0:
            return 0.0
        return self.cacheable_tokens / self.submitted_tokens

    def as_dict(self) -> dict[str, int | float]:
        return {
            "turns": self.turns,
            "cacheable_tokens": self.cacheable_tokens,
            "submitted_tokens": self.submitted_tokens,
            "eligible_fraction": self.eligible_fraction,
        }
