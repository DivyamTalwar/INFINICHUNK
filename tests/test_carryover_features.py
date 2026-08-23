from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from infinichunk_ext.carryover import CarryoverBenchmark, ContinuityVerifier, SalienceCarryoverSelector


class CarryoverFeatureTests(unittest.TestCase):
    def test_salience_selector_preserves_budget_head_tail_and_order(self):
        selector = SalienceCarryoverSelector(budget=5, keep_first=1, keep_last=1)
        selection = selector.select(range(10), [0, 1, 9, 2, 8, 3, 7, 4, 5, 0])
        self.assertEqual(selection.source_indices, (0, 2, 4, 6, 9))
        self.assertEqual(selection.token_ids, (0, 2, 4, 6, 9))

    def test_continuity_verifier_flags_explicit_conflicts(self):
        issues = ContinuityVerifier().compare("x = 4; goal: prove", "x = 5; goal: prove")
        self.assertEqual([(issue.symbol, issue.previous_value, issue.current_value) for issue in issues], [("x", "4", "5")])

    def test_benchmark_emits_machine_readable_receipts(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "receipts.jsonl"
            benchmark = CarryoverBenchmark(output)
            policies = {"salience": SalienceCarryoverSelector(2)}
            receipts = benchmark.run(
                [("case", [1, 2, 3], [0.1, 0.9, 0.2])],
                policies,
                lambda _, selection: float(2 in selection.token_ids),
            )
            self.assertEqual(receipts[0].utility, 1.0)
            self.assertTrue(output.read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
