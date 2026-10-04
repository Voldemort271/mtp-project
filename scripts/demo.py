"""Run a clearly synthetic end-to-end train/predict/rank smoke demo."""

from __future__ import annotations

import pandas as pd

from _common import PROJECT_ROOT  # Adds the repository root for direct execution.


def _synthetic_training_data() -> pd.DataFrame:
    rows = []
    cases = [
        # Expected-return mutant: failing tests become passing.
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "fail", "pass"),
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "pass", "pass"),
        ("MOR", "*", "+", 4, "BinaryOperation", 0, "pass", "pass"),
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "fail", "pass"),
        # Normalize mutant: the covered passing test becomes failing.
        ("MOR", "/", "*", 2, "Return", 0, "fail", "fail"),
        ("MOR", "/", "*", 2, "Return", 0, "pass", "pass"),
        ("MOR", "/", "*", 2, "Return", 1, "pass", "fail"),
        ("MOR", "/", "*", 2, "Return", 0, "fail", "fail"),
    ]
    for version in range(8):
        for mutation, operator, replacement, depth, parent, covered, baseline, mutant in cases:
            rows.append(
                {
                    "source_version": f"synthetic-version-{version}",
                    "mutation_type": mutation,
                    "mutated_operator": operator,
                    "replacement_operator": replacement,
                    "ast_depth": depth,
                    "parent_node_type": parent,
                    "test_covers_mutation": covered,
                    "baseline_test_result": baseline,
                    "mutant_test_result": mutant,
                }
            )
    from ml_pmt import add_outcome_change_labels

    return add_outcome_change_labels(pd.DataFrame(rows))


def main() -> int:
    from ml_pmt import OutcomeChangeModel, predict_and_rank

    training = _synthetic_training_data()
    model = OutcomeChangeModel(n_estimators=50).fit(training)
    baseline = {"T1": "fail", "T2": "pass", "T3": "pass", "T4": "fail"}
    candidate_rows = []
    for line, depth, parent, coverage in (
        (3, 4, "BinaryOperation", [1, 1, 0, 1]),
        (6, 2, "Return", [0, 0, 1, 0]),
    ):
        for (test_id, baseline_result), covered in zip(
            baseline.items(), coverage, strict=True
        ):
            candidate_rows.append(
                {
                    "line_number": line,
                    "test_id": test_id,
                    "mutation_type": "MOR",
                    "mutated_operator": "*" if line == 3 else "/",
                    "replacement_operator": "+" if line == 3 else "*",
                    "ast_depth": depth,
                    "parent_node_type": parent,
                    "test_covers_mutation": covered,
                    "baseline_test_result": baseline_result,
                }
            )

    ranking = predict_and_rank(model, pd.DataFrame(candidate_rows), baseline)
    print("Synthetic smoke demo only; this does not measure generalization.")
    print(ranking.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
