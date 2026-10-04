"""Compare model configurations on leave-one-bug-out localization quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml_pmt import OutcomeChangeModel, predict_and_rank
from scripts.evaluate_localization import rank_metrics

CONFIGS = [
    ("rf-onehot", {"estimator": "rf", "categorical_encoding": "onehot"}),
    ("rf-ordinal", {"estimator": "rf", "categorical_encoding": "ordinal"}),
    ("gbt-ordinal", {"estimator": "gbt", "categorical_encoding": "ordinal"}),
    ("gbt-onehot", {"estimator": "gbt", "categorical_encoding": "onehot"}),
]


def evaluate_config(
    pairs: pd.DataFrame,
    config_name: str,
    kwargs: dict,
    *,
    random_state: int = 42,
) -> pd.DataFrame:
    faults = pd.read_csv(ROOT / "data" / "fault_labels.csv")
    records = []
    for variant_id, group in faults.groupby("source_version"):
        variant_id = str(variant_id)
        fault_lines = sorted(int(line) for line in group["fault_line"])
        bug_key = variant_id.removesuffix("-fixed")
        train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
        candidates = pairs[pairs["source_version"] == variant_id]
        model = OutcomeChangeModel(
            random_state=random_state, **kwargs
        ).fit(train)
        baseline = dict(
            zip(candidates["test_id"], candidates["baseline_test_result"], strict=True)
        )
        ranking = predict_and_rank(model, candidates, baseline)
        metrics = rank_metrics(ranking, fault_lines)
        metrics["config"] = config_name
        metrics["source_version"] = variant_id
        records.append(metrics)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "data" / "pairs.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    frames = []
    for name, kwargs in CONFIGS:
        result = evaluate_config(pairs, name, kwargs, random_state=args.seed)
        frames.append(result)
        agg = result.groupby("config").agg(
            top1=("top1", "mean"),
            top5=("top5", "mean"),
            mrr=("mrr", "mean"),
            exam=("exam", "mean"),
        )
        print(f"\n=== {name} ===")
        per = result.set_index("source_version")[["rank"]]
        print(per.to_string())
        print(agg.round(3).to_string())

    all_results = pd.concat(frames, ignore_index=True)
    print("\n=== Summary (mean across 7 bugs) ===")
    summary = all_results.groupby("config").agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        mrr=("mrr", "mean"),
        exam=("exam", "mean"),
    ).round(3)
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())