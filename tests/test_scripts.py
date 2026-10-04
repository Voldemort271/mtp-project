import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_train_evaluate_and_localize_scripts(tmp_path):
    training_rows = []
    patterns = [
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "fail", "pass"),
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "pass", "pass"),
        ("MOR", "*", "+", 4, "BinaryOperation", 0, "pass", "pass"),
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "fail", "pass"),
        ("MOR", "/", "*", 2, "Return", 0, "fail", "fail"),
        ("MOR", "/", "*", 2, "Return", 0, "pass", "pass"),
        ("MOR", "/", "*", 2, "Return", 1, "pass", "fail"),
        ("MOR", "/", "*", 2, "Return", 0, "fail", "fail"),
    ]
    for version in range(8):
        for mutation, operator, replacement, depth, parent, covered, baseline, mutant in patterns:
            training_rows.append(
                {
                    "source_version": f"version-{version}",
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
    training_path = tmp_path / "pairs.csv"
    pd.DataFrame(training_rows).to_csv(training_path, index=False)

    candidates = []
    for line, depth, parent, coverages in (
        (3, 4, "BinaryOperation", [1, 1, 0, 1]),
        (6, 2, "Return", [0, 0, 1, 0]),
    ):
        for (test_id, baseline), covered in zip(
            (("T1", "fail"), ("T2", "pass"), ("T3", "pass"), ("T4", "fail")),
            coverages,
            strict=True,
        ):
            candidates.append(
                {
                    "line_number": line,
                    "test_id": test_id,
                    "mutation_type": "MOR",
                    "mutated_operator": "*" if line == 3 else "/",
                    "replacement_operator": "+" if line == 3 else "*",
                    "ast_depth": depth,
                    "parent_node_type": parent,
                    "test_covers_mutation": covered,
                    "baseline_test_result": baseline,
                }
            )
    candidate_path = tmp_path / "candidates.csv"
    pd.DataFrame(candidates).to_csv(candidate_path, index=False)
    baseline_path = tmp_path / "baseline.csv"
    pd.DataFrame(
        [("T1", "fail"), ("T2", "pass"), ("T3", "pass"), ("T4", "fail")],
        columns=["test_id", "result"],
    ).to_csv(baseline_path, index=False)

    model_path = tmp_path / "model.joblib"
    _run_script(
        "train_model.py",
        "--pairs",
        str(training_path),
        "--output",
        str(model_path),
    )
    assert model_path.exists()

    evaluation = _run_script("evaluate_model.py", "--pairs", str(training_path))
    assert "Brier score:" in evaluation.stdout

    ranking_path = tmp_path / "ranking.csv"
    _run_script(
        "localize.py",
        "--model",
        str(model_path),
        "--candidates",
        str(candidate_path),
        "--baseline-tests",
        str(baseline_path),
        "--output",
        str(ranking_path),
    )
    ranking = pd.read_csv(ranking_path)
    assert set(ranking["line_number"]) == {3, 6}
    assert ranking["suspiciousness"].between(0, 1).all()


def test_mlpmt_entry_point_reports_status_and_runs_modes():
    status = _run_script("mlpmt.py", "status")
    assert "pairs.csv:" in status.stdout
    assert "source versions" in status.stdout

    help_result = _run_script("mlpmt.py", "--help")
    assert "evaluate" in help_result.stdout
    assert "baselines" in help_result.stdout
    assert "localize" in help_result.stdout


def _run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
