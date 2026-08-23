"""Original carryover policies and cross-chunk continuity checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class CarryoverSelection:
    token_ids: tuple[int, ...]
    source_indices: tuple[int, ...]
    policy: str


class SalienceCarryoverSelector:
    """Select a bounded, ordered carryover from externally learned salience.

    The selector is scorer-agnostic: a learned value/state-reconstruction model
    can supply one score per token while this class enforces budget, stable
    ordering, and mandatory prompt/head preservation.
    """

    def __init__(self, budget: int, keep_first: int = 0, keep_last: int = 0):
        if budget < 1 or keep_first < 0 or keep_last < 0:
            raise ValueError("carryover budget must be positive and keep counts non-negative")
        if keep_first + keep_last > budget:
            raise ValueError("mandatory head/tail exceeds carryover budget")
        self.budget = budget
        self.keep_first = keep_first
        self.keep_last = keep_last

    def select(self, token_ids: Sequence[int], salience: Sequence[float]) -> CarryoverSelection:
        if len(token_ids) != len(salience):
            raise ValueError("one salience score is required per token")
        if len(token_ids) <= self.budget:
            indices = tuple(range(len(token_ids)))
            return CarryoverSelection(tuple(token_ids), indices, "salience")

        mandatory = set(range(self.keep_first))
        if self.keep_last:
            mandatory.update(range(len(token_ids) - self.keep_last, len(token_ids)))
        candidates = (index for index in range(len(token_ids)) if index not in mandatory)
        ranked = sorted(candidates, key=lambda index: (-salience[index], index))
        selected = sorted(mandatory.union(ranked[: self.budget - len(mandatory)]))
        return CarryoverSelection(
            tuple(token_ids[index] for index in selected),
            tuple(selected),
            "salience",
        )


@dataclass(frozen=True)
class ContinuityIssue:
    symbol: str
    previous_value: str
    current_value: str
    issue_type: str = "contradiction"


class ContinuityVerifier:
    """Detect explicit variable/value contradictions across chunk summaries."""

    ASSIGNMENT = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s*(?:=|is|:)\s*([^,;\n.]+)")

    def extract_assignments(self, text: str) -> dict[str, str]:
        return {
            match.group(1).casefold(): match.group(2).strip().casefold()
            for match in self.ASSIGNMENT.finditer(text)
        }

    def compare(self, previous: str, current: str) -> list[ContinuityIssue]:
        prior = self.extract_assignments(previous)
        now = self.extract_assignments(current)
        return [
            ContinuityIssue(symbol, prior[symbol], value)
            for symbol, value in sorted(now.items())
            if symbol in prior and prior[symbol] != value
        ]


@dataclass(frozen=True)
class BenchmarkReceipt:
    case_id: str
    policy: str
    tokens_available: int
    tokens_kept: int
    utility: float


class CarryoverBenchmark:
    """Compare carryover policies under one token budget and scorer."""

    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        cases: Iterable[tuple[str, Sequence[int], Sequence[float]]],
        policies: dict[str, SalienceCarryoverSelector],
        utility: Callable[[str, CarryoverSelection], float],
    ) -> list[BenchmarkReceipt]:
        receipts: list[BenchmarkReceipt] = []
        for case_id, tokens, scores in cases:
            for name, policy in policies.items():
                selection = policy.select(tokens, scores)
                receipts.append(
                    BenchmarkReceipt(case_id, name, len(tokens), len(selection.token_ids), utility(case_id, selection))
                )
        self.output_path.write_text(
            "".join(json.dumps(asdict(receipt), sort_keys=True) + "\n" for receipt in receipts),
            encoding="utf-8",
        )
        return receipts
