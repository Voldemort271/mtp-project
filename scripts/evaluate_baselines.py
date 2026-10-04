"""Compare ML-PMT against SBFL and real-execution MBFL baselines."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.pytest_runner import coverage_by_test_file, run_pytest
from ml_pmt import OutcomeChangeModel, predict_and_rank
from ml_pmt.ranker import line_aggregates
from scripts.build_bugs import BUGS, BUGS_ROOT, WORK, make_sandbox, select_tests
from scripts.evaluate_localization import honest_two_stage_ranking, rank_metrics
from scripts.build_bugs import BUGS, WORK
from scripts.evaluate_localization import rank_metrics

BUG_SPECS = {f"{spec['project']}-{spec['bug']}": spec for spec in BUGS}


def variant_environment(variant_id: str) -> tuple[Path, Path, list[str]]:
    """Return (checkout, venv python, source packages) for a buggy variant."""
    spec = BUG_SPECS[variant_id]
    checkout = WORK / f"{spec['project']}_{spec['bug']}" / spec["project"]
    python = WORK / spec["venv"] / "venv" / "bin" / "python"
    return checkout, python, spec["packages"]


def predict_ranking(
    pairs: pd.DataFrame,
    variant_id: str,
    random_state: int,
) -> pd.DataFrame:
    """Rank a held-out variant with the trained model (leave-one-bug-out)."""
    bug_key = variant_id.removesuffix("-fixed")
    train = pairs[~pairs["source_version"].str.removesuffix("-fixed").eq(bug_key)]
    candidates = pairs[pairs["source_version"] == variant_id]
    model = OutcomeChangeModel(random_state=random_state).fit(train)
    baseline = dict(
        zip(candidates["test_id"], candidates["baseline_test_result"], strict=True)
    )
    return predict_and_rank(model, candidates, baseline)


def metallaxis_ranking(pairs: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    """Rank lines using real (executed) outcome-change labels."""
    variant = pairs[pairs["source_version"] == variant_id]
    baseline = dict(
        zip(variant["test_id"], variant["baseline_test_result"], strict=True)
    )
    failing = {test_id for test_id, result in baseline.items() if result == "fail"}
    passing = {test_id for test_id, result in baseline.items() if result == "pass"}
    total_failing = len(failing)

    changed = (
        variant.groupby(["line_number", "test_id"], as_index=False)["outcome_changed"]
        .max()
    )
    rows = []
    for line, group in changed.groupby("line_number"):
        changed_tests = set(group[group["outcome_changed"] == 1]["test_id"])
        f_killed = len(changed_tests & failing)
        p_killed = len(changed_tests & passing)
        denominator = np.sqrt(total_failing * (f_killed + p_killed))
        score = f_killed / denominator if denominator > 0 else 0.0
        rows.append(
            {
                "line_number": int(line),
                "f_killed": float(f_killed),
                "p_killed": float(p_killed),
                "suspiciousness": float(score),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def sbfl_ranking(
    checkout: Path,
    python: Path,
    test_args: list[str],
    fault_files: list[str],
    source_packages: list[str],
    baseline: dict[str, str],
) -> dict[str, pd.DataFrame]:
    """Rank covered lines in the fault files with Ochiai and Tarantula."""
    coverage = coverage_by_test_file(
        checkout, test_args, python, source_packages=source_packages
    )
    failing = [test_id for test_id, result in baseline.items() if result == "fail"]
    passing = [test_id for test_id, result in baseline.items() if result == "pass"]

    # (line, file) -> set of tests covering it
    covered_lines: dict[tuple[int, str], set[str]] = {}
    for test_id, files in coverage.items():
        for filename, lines in files.items():
            if any(filename.endswith(fault_file) for fault_file in fault_files):
                for line in lines:
                    covered_lines.setdefault((line, filename), set()).add(test_id)

    rows = []
    for (line, filename), covering in covered_lines.items():
        a = len(set(failing) & covering)
        b = len(set(passing) & covering)
        if a + b == 0:
            continue
        c = len(failing) - a
        e = len(passing) - b
        ochiai_den = np.sqrt((a + b) * (a + c))
        ochiai = a / ochiai_den if ochiai_den > 0 else 0.0
        positive = a / (a + c) if a + c > 0 else 0.0
        negative = b / (b + e) if b + e > 0 else 0.0
        tarantula_den = positive + negative
        tarantula = positive / tarantula_den if tarantula_den > 0 else 0.0
        rows.append(
            {
                "line_number": line,
                "file": filename,
                "a": a,
                "b": b,
                "ochiai": float(ochiai),
                "tarantula": float(tarantula),
            }
        )
    frame = pd.DataFrame(rows)
    rankings = {}
    for method, column in (("ochiai", "ochiai"), ("tarantula", "tarantula")):
        if frame.empty:
            rankings[method] = pd.DataFrame(
                columns=["line_number", "suspiciousness"]
            )
            continue
        ranking = frame.loc[:, ["line_number", column]].rename(
            columns={column: "suspiciousness"}
        )
        rankings[method] = ranking.sort_values(
            ["suspiciousness", "line_number"],
            ascending=[False, True],
            ignore_index=True,
        )
    rankings["_spectrum"] = frame
    return rankings


def hybrid_ranking(
    pairs: pd.DataFrame,
    variant_id: str,
    *,
    checkout: Path,
    python: Path,
    test_args: list[str],
    fault_files: list[str],
    source_packages: list[str],
    baseline: dict[str, str],
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank the full covered line universe with max(SBFL, mutation score).

    Mutation scores come from the trained pair model on mutant lines; lines
    without mutants fall back to their Ochiai score. The max fusion cannot be
    worse than SBFL alone.
    """
    spectrum = sbfl_ranking(
        checkout, python, test_args, fault_files, source_packages, baseline
    )["_spectrum"]
    if spectrum.empty:
        return pd.DataFrame(columns=["line_number", "suspiciousness"])

    candidates = pairs[pairs["source_version"] == variant_id]
    model = OutcomeChangeModel(random_state=random_state).fit(
        pairs[
            ~pairs["source_version"].str.removesuffix("-fixed")
            .eq(variant_id.removesuffix("-fixed"))
        ]
    )
    probs = model.predict_probabilities(candidates)
    aggregates = line_aggregates(candidates, probs, collapse="max")
    failing_count = sum(1 for result in baseline.values() if result == "fail")
    mutation = aggregates["f_killed"] / np.sqrt(
        np.maximum(failing_count, 1)
        * (aggregates["f_killed"] + aggregates["p_killed"])
    )
    mutation_by_line = dict(zip(aggregates["line_number"], mutation))

    spectrum["mut_score"] = spectrum["line_number"].map(mutation_by_line).fillna(0.0)

    def _normalize(series: pd.Series) -> pd.Series:
        spread = series.max() - series.min()
        return (series - series.min()) / spread if spread > 0 else series * 0.0

    spectrum["hybrid"] = np.maximum(
        _normalize(spectrum["ochiai"]), _normalize(spectrum["mut_score"])
    )
    return spectrum.loc[:, ["line_number", "hybrid"]].rename(
        columns={"hybrid": "suspiciousness"}
    ).sort_values(
        ["suspiciousness", "line_number"], ascending=[False, True], ignore_index=True
    )


def _project_run_times(pairs: pd.DataFrame) -> dict[str, float]:
    """Measure the wall-clock of one full test-suite run per project."""
    run_times: dict[str, float] = {}
    for spec in BUGS:
        project = spec["project"]
        if project in run_times:
            continue
        checkout, python, _packages = variant_environment(
            f"{project}-{spec['bug']}"
        )
        bug_dir = BUGS_ROOT / project / "bugs" / str(spec["bug"])
        sandbox = make_sandbox(checkout, f"{project}_rt", project)
        tests, _failing = select_tests(sandbox, bug_dir, python)
        t0 = time.perf_counter()
        run_pytest(sandbox, tests, python)
        run_times[project] = time.perf_counter() - t0
    return run_times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=ROOT / "data" / "pairs.csv",
    )
    parser.add_argument(
        "--faults",
        type=Path,
        default=ROOT / "data" / "fault_labels.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = pd.read_csv(args.pairs)
    faults = pd.read_csv(args.faults)
    run_times = _project_run_times(pairs)

    records = []
    for variant_id, group in faults.groupby("source_version"):
        variant_id = str(variant_id)
        fault_lines = sorted(int(line) for line in group["fault_line"])
        variant = pairs[pairs["source_version"] == variant_id]
        baseline = dict(
            zip(variant["test_id"], variant["baseline_test_result"], strict=True)
        )
        tests = sorted(set(variant["test_id"]))
        fault_files = sorted(set(variant["file"]))

        rankings = {
            "ML-PMT (predicted)": predict_ranking(pairs, variant_id, args.seed),
            "ML-PMT (two-stage)": honest_two_stage_ranking(
                pairs, faults, variant_id, args.seed
            ),
            "MBFL-real (Metallaxis)": metallaxis_ranking(pairs, variant_id),
        }
        project = variant_id.split("-")[0]
        bug = variant_id.split("-")[1]
        checkout, python, packages = variant_environment(variant_id)
        for method, ranking in sbfl_ranking(
            checkout, python, tests, fault_files, packages, baseline
        ).items():
            if method.startswith("_"):
                continue
            rankings[f"SBFL-{method}"] = ranking
        rankings["Hybrid (SBFL+pred)"] = hybrid_ranking(
            pairs,
            variant_id,
            checkout=checkout,
            python=python,
            test_args=tests,
            fault_files=fault_files,
            source_packages=packages,
            baseline=baseline,
            random_state=args.seed,
        )

        for method, ranking in rankings.items():
            metrics = rank_metrics(ranking, fault_lines)
            metrics.update(
                {
                    "method": method,
                    "source_version": variant_id,
                    "num_ranked": len(ranking),
                }
            )
            _attach_cost(metrics, method, variant_id, pairs, run_times)
            records.append(metrics)

    results = pd.DataFrame(records)
    wide = results.pivot_table(
        index="source_version",
        columns="method",
        values="rank",
        aggfunc="first",
    )
    print("=== Fault-line rank per variant ===")
    print(wide.to_string())
    print()
    print("=== Aggregate metrics ===")
    agg = results.groupby("method").agg(
        top1=("top1", "mean"),
        top5=("top5", "mean"),
        top10=("top10", "mean"),
        mrr=("mrr", "mean"),
        rank=("rank", "mean"),
        exam=("exam", "mean"),
        exam_at_10=("exam_at_10", "mean"),
        exam_at_20=("exam_at_20", "mean"),
        executions=("executions", "mean"),
        time_s=("time_s", "mean"),
    ).reindex(
        ["Hybrid (SBFL+pred)", "ML-PMT (predicted)", "ML-PMT (two-stage)",
         "MBFL-real (Metallaxis)", "SBFL-ochiai", "SBFL-tarantula"]
    ).round(3)
    print(agg.to_string())
    print("\n=== Inference cost (test-suite executions per bug) ===")
    _print_cost(pairs)
    return 0


def _attach_cost(
    metrics: dict,
    method: str,
    variant_id: str,
    pairs: pd.DataFrame,
    run_times: dict[str, float],
) -> None:
    """Add test-execution time and suite-run count for a method+bug."""
    project = variant_id.split("-")[0]
    n_mutants = int(
        pairs[pairs["source_version"] == variant_id]["mutant_id"].nunique()
    )
    t_run = run_times.get(project, 1.0)
    if "MBFL-real" in method:
        executions = n_mutants
    elif method.startswith("SBFL"):
        executions = 1
    else:  # ML-PMT and hybrid
        executions = 2
    metrics["executions"] = executions
    metrics["time_s"] = executions * t_run


def _print_cost(pairs: pd.DataFrame) -> None:
    """Report the execution cost of each method per localized bug."""
    per_bug = []
    for variant_id in pairs["source_version"].unique():
        variant = pairs[pairs["source_version"] == variant_id]
        n_tests = variant["test_id"].nunique()
        n_mutants = variant["mutant_id"].nunique()
        per_bug.append((variant_id, n_mutants, n_tests))
    mean_mutants = sum(m for _, m, _ in per_bug) / len(per_bug)
    print(
        f"MBFL-real (Metallaxis):  {mean_mutants:.0f} mutant executions/bug "
        f"(each runs the full test suite once)"
    )
    print("ML-PMT / Hybrid:         2 test runs/bug (1 baseline + 1 coverage), "
          "no mutant execution")
    print("SBFL:                    1 coverage run/bug")
    print(
        f"Execution savings vs MBFL-real: "
        f"~{mean_mutants / 2:.0f}x fewer executions per bug for ML-PMT"
    )


if __name__ == "__main__":
    raise SystemExit(main())