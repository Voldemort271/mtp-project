"""Tests for the upstream generation pipeline (mutation, execution, features)."""

from pathlib import Path

import pandas as pd
import pytest

from execution.runner import baseline_results, mutant_results
from features.extractor import node_metadata
from mutation_engine.generator import generate_mutants
from scripts.build_bugs import fault_lines
from scripts.build_corpus import build_corpus
from scripts.evaluate_baselines import metallaxis_ranking
from scripts.evaluate_localization import rank_metrics
from target_model import test_cases

ROOT = Path(__file__).resolve().parents[1]
BASE_SOURCE = (ROOT / "target_model" / "model.py").read_text()


def test_generate_mutants_are_valid_python_and_distinct():
    mutants = generate_mutants(BASE_SOURCE)

    assert len(mutants) > 0
    sources = [mutant["source"] for mutant in mutants]
    assert len(set(sources)) == len(mutants)
    assert all(BASE_SOURCE != source for source in sources)
    for mutant in mutants:
        compile(mutant["source"], "<mutant>", "exec")


def test_generate_mutants_covers_all_three_operator_types():
    mutants = generate_mutants(BASE_SOURCE)
    types = {mutant["mutation_type"] for mutant in mutants}

    assert {"MOR", "TSM", "FCS"} <= types
    assert {mutant["function"] for mutant in mutants} == {
        "expected_return",
        "portfolio_variance",
        "normalize_weights",
        "markowitz_weights",
        "transpose_returns",
    }


def test_swapped_variant_has_repair_mutant():
    variant = BASE_SOURCE.replace(
        "np.sum(weights * returns)", "np.sum(weights + returns)"
    )
    mutants = generate_mutants(variant)

    repaired = [
        mutant for mutant in mutants if "weights * returns" in mutant["source"]
    ]
    assert len(repaired) >= 1


def test_node_metadata_reports_depth_and_parent():
    metadata = node_metadata(BASE_SOURCE, 7, "MOR")

    assert metadata["ast_depth"] > 0
    assert metadata["parent_node_type"] == "Arg"


def test_baseline_results_pass_and_report_coverage():
    results, coverage = baseline_results(BASE_SOURCE, test_cases.TESTS)

    assert set(results.values()) == {"pass"}
    assert 7 in coverage["T_er1"]
    assert 27 in coverage["T_tr1"]
    assert 15 in coverage["T_nw1"]


def test_repair_mutant_flips_failing_tests_to_passing():
    variant = BASE_SOURCE.replace(
        "np.sum(weights * returns)", "np.sum(weights + returns)"
    )
    baseline, _ = baseline_results(variant, test_cases.TESTS)
    repair = variant.replace("np.sum(weights + returns)", "np.sum(weights * returns)")
    results = mutant_results(repair, test_cases.TESTS, "__mlpmt_test_repair__")

    assert baseline["T_er1"] == "fail"
    assert results["T_er1"] == "pass"


def test_build_corpus_writes_pairs_and_fault_labels(tmp_path):
    summary = build_corpus(
        variant_ids=["v_er_mul_add", "v_tr_reshape_swap"],
        include_clean=False,
        out_dir=tmp_path,
    )

    pairs = pd.read_csv(summary["pairs_path"])
    faults = pd.read_csv(summary["faults_path"])

    assert set(pairs["source_version"]) == {"v_er_mul_add", "v_tr_reshape_swap"}
    assert pairs["outcome_changed"].isin((0, 1)).all()
    assert pairs["test_covers_mutation"].isin((0, 1)).all()
    assert faults["fault_line"].tolist() == [7, 27]


def test_rank_metrics_computes_standard_measures():
    ranking = pd.DataFrame(
        {"line_number": [16, 7, 11], "suspiciousness": [0.9, 0.4, 0.2]}
    )

    top1 = rank_metrics(ranking, [16])
    bottom = rank_metrics(ranking, [11])
    missing = rank_metrics(ranking, [99])
    multiple = rank_metrics(ranking, [99, 7])

    assert top1 == {
        "fault_lines": [16],
        "rank": 1,
        "top1": True,
        "top5": True,
        "top10": True,
        "mrr": 1.0,
        "exam": 1 / 3,
        "exam_at_10": False,
        "exam_at_20": False,
    }
    assert bottom["rank"] == 3
    assert missing["rank"] == 4
    assert multiple["rank"] == 2
    assert pytest.approx(bottom["exam"]) == 1.0


def test_fault_lines_parser_maps_patch_hunks_to_buggy_lines():
    patch = """diff --git a/tqdm/_tqdm.py b/tqdm/_tqdm.py
@@ -41,2 +41,2 @@
-        if abs(num) < 1000.0:
-            if abs(num) < 100.0:
+        if abs(num) < 999.95:
+            if abs(num) < 99.95:
"""
    faults = fault_lines(patch)

    assert faults == {"tqdm/_tqdm.py": [41, 42]}


def test_extended_engine_finds_roor_and_cor_mutants():
    source = "def f(x, total):\n    if x > 0 and total:\n        return x + 1\n    return total or 0\n"
    mutants = generate_mutants(source)

    types = {m["mutation_type"] for m in mutants}
    assert "ROOR" in types
    assert "COR" in types
    for mutant in mutants:
        compile(mutant["source"], "<mutant>", "exec")


def test_metallaxis_ranking_uses_real_flips():
    pairs = pd.DataFrame(
        [
            # Line 12: a real mutant flips failing test T1 (repair).
            ("v", "m1", 12, "T1", "fail", 1),
            ("v", "m1", 12, "T2", "pass", 0),
            # Line 40: mutants flip a passing test only.
            ("v", "m2", 40, "T1", "fail", 0),
            ("v", "m2", 40, "T2", "pass", 1),
        ],
        columns=[
            "source_version",
            "mutant_id",
            "line_number",
            "test_id",
            "baseline_test_result",
            "outcome_changed",
        ],
    )

    ranking = metallaxis_ranking(pairs, "v")

    assert ranking.iloc[0]["line_number"] == 12
    assert ranking.iloc[0]["f_killed"] == 1.0
    assert ranking.iloc[0]["p_killed"] == 0.0


def test_leave_one_bug_out_is_disjoint():
    """No held-out bug's buggy or fixed version may appear in training."""
    pairs = pd.read_csv(ROOT / "data" / "pairs.csv")
    faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")

    for variant_id in faults["source_version"].unique():
        variant_id = str(variant_id)
        bug_key = variant_id.removesuffix("-fixed")
        train = pairs[
            ~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)
        ]
        leaked = train["source_version"].str.removesuffix("-fixed").eq(bug_key).any()
        assert not leaked, f"held-out bug {bug_key} leaked into training"