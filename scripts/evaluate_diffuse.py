"""Compare SBFL vs real-MBFL vs ML-PMT vs hybrid on the diffuse-coverage set.

SBFL is expected to be weak here because every test covers the fault line
(diffuse coverage); the mutation signal should find the faults.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel
from ml_pmt.ranker import line_aggregates
from scripts.evaluate_localization import rank_metrics


def sbfl_ranking(pairs: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    """Ochiai over the covered lines from the corpus coverage data."""
    variant = pairs[pairs["source_version"] == variant_id]
    baseline = dict(
        zip(variant["test_id"], variant["baseline_test_result"], strict=True)
    )
    failing = {t for t, r in baseline.items() if r == "fail"}
    passing = {t for t, r in baseline.items() if r == "pass"}
    rows = []
    for line, group in variant.groupby("line_number"):
        covered = set(group[group["test_covers_mutation"] == 1]["test_id"])
        a = len(covered & failing)
        b = len(covered & passing)
        if a + b == 0:
            continue
        c = len(failing) - a
        denom = np.sqrt((a + b) * (a + c))
        ochiai = a / denom if denom > 0 else 0.0
        rows.append({"line_number": int(line), "suspiciousness": float(ochiai)})
    return pd.DataFrame(rows).sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def metallaxis_ranking(pairs: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    variant = pairs[pairs["source_version"] == variant_id]
    baseline = dict(
        zip(variant["test_id"], variant["baseline_test_result"], strict=True)
    )
    failing = {t for t, r in baseline.items() if r == "fail"}
    passing = {t for t, r in baseline.items() if r == "pass"}
    rows = []
    for line, group in variant.groupby("line_number"):
        changed = set(group[group["outcome_changed"] == 1]["test_id"])
        f_killed = len(changed & failing)
        p_killed = len(changed & passing)
        denom = np.sqrt(len(failing) * (f_killed + p_killed))
        score = f_killed / denom if denom > 0 else 0.0
        rows.append({"line_number": int(line), "suspiciousness": float(score)})
    return pd.DataFrame(rows).sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def mlpmt_ranking(
    bugsinpy_pairs: pd.DataFrame,
    diffuse_pairs: pd.DataFrame,
    variant_id: str,
) -> pd.DataFrame:
    model = OutcomeChangeModel(random_state=42).fit(bugsinpy_pairs)
    candidates = diffuse_pairs[diffuse_pairs["source_version"] == variant_id]
    aggregates = line_aggregates(candidates, model.predict_probabilities(candidates))
    failing_count = int(
        (candidates["baseline_test_result"] == "fail").sum()
        / candidates["test_id"].nunique()
        or 1
    )
    score = aggregates["f_killed"] / np.sqrt(
        np.maximum(failing_count, 1) * (aggregates["f_killed"] + aggregates["p_killed"])
    )
    return pd.DataFrame(
        {"line_number": aggregates["line_number"], "suspiciousness": score}
    ).sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def mlpmt_loo_ranking(
    diffuse_pairs: pd.DataFrame, variant_id: str
) -> pd.DataFrame:
    """Leave-one-out within the diffuse set (fair training distribution)."""
    train = diffuse_pairs[diffuse_pairs["source_version"] != variant_id]
    model = OutcomeChangeModel(random_state=42).fit(train)
    candidates = diffuse_pairs[diffuse_pairs["source_version"] == variant_id]
    aggregates = line_aggregates(candidates, model.predict_probabilities(candidates))
    failing_count = int(
        (candidates["baseline_test_result"] == "fail").sum()
        / candidates["test_id"].nunique()
        or 1
    )
    score = aggregates["f_killed"] / np.sqrt(
        np.maximum(failing_count, 1) * (aggregates["f_killed"] + aggregates["p_killed"])
    )
    return pd.DataFrame(
        {"line_number": aggregates["line_number"], "suspiciousness": score}
    ).sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def hybrid_ranking(
    bugsinpy_pairs, diffuse_pairs, variant_id, sbfl: pd.DataFrame
) -> pd.DataFrame:
    ml = mlpmt_loo_ranking(diffuse_pairs, variant_id)
    merged = sbfl.merge(ml, on="line_number", suffixes=("_s", "_m"))
    norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    merged["suspiciousness"] = np.maximum(
        norm(merged["suspiciousness_s"]), norm(merged["suspiciousness_m"])
    )
    return merged.loc[:, ["line_number", "suspiciousness"]].sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def main() -> int:
    diffuse = pd.read_csv(ROOT / "data" / "diffuse" / "pairs.csv")
    faults = pd.read_csv(ROOT / "data" / "diffuse" / "fault_labels.csv")
    bugsinpy = pd.read_csv(ROOT / "data" / "pairs.csv")

    records = []
    for variant_id in map(str, faults["source_version"].unique()):
        fault_line = sorted(int(x) for x in faults[faults["source_version"] == variant_id]["fault_line"])
        sbfl = sbfl_ranking(diffuse, variant_id)
        mbfl = metallaxis_ranking(diffuse, variant_id)
        ml_bugs = mlpmt_ranking(bugsinpy, diffuse, variant_id)
        ml_loo = mlpmt_loo_ranking(diffuse, variant_id)
        hyb = hybrid_ranking(bugsinpy, diffuse, variant_id, sbfl)
        for name, ranking in (
            ("SBFL-ochiai", sbfl),
            ("MBFL-real", mbfl),
            ("ML-PMT (trained on BugsInPy)", ml_bugs),
            ("ML-PMT (diffuse LOO)", ml_loo),
            ("Hybrid (SBFL+diffuse LOO)", hyb),
        ):
            metrics = rank_metrics(ranking, fault_line)
            metrics["method"] = name
            metrics["source_version"] = variant_id
            records.append(metrics)

    results = pd.DataFrame(records)
    agg = results.groupby("method").agg(
        top1=("top1", "mean"),
        mrr=("mrr", "mean"),
        rank=("rank", "mean"),
        exam=("exam", "mean"),
    ).reindex(
        ["SBFL-ochiai", "MBFL-real", "ML-PMT (trained on BugsInPy)",
         "ML-PMT (diffuse LOO)", "Hybrid (SBFL+diffuse LOO)"]
    )
    print("=== Diffuse-coverage benchmark (SBFL-weak setting) ===")
    print(agg.round(3).to_string())
    print()
    print(results.pivot_table(index="source_version", columns="method", values="rank", aggfunc="first").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())