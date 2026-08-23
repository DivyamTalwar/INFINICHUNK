from pathlib import Path

from infinichunk_ext.carryover import CarryoverBenchmark, SalienceCarryoverSelector


cases = [("demo", list(range(16)), [float(index % 5) for index in range(16)])]
policies = {
    "head-tail": SalienceCarryoverSelector(budget=8, keep_first=4, keep_last=4),
    "salience": SalienceCarryoverSelector(budget=8, keep_first=1, keep_last=1),
}
receipts = CarryoverBenchmark(Path("results/carryover_demo.jsonl")).run(
    cases,
    policies,
    lambda _, selection: sum(selection.token_ids) / max(1, len(selection.token_ids)),
)
print(receipts)
