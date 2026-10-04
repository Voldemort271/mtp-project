"""Build the diffuse-coverage benchmark corpus.

Each diffuse module's fault sits on a line covered by every test (diffuse
coverage), so coverage-based (SBFL) localization cannot discriminate it, while
a mutant on that line flips the failing tests. Produces data/diffuse/pairs.csv
and data/diffuse/fault_labels.csv in the same schema as the BugsInPy corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.runner import baseline_results, mutant_results
from features.extractor import node_metadata
from mutation_engine.generator import generate_mutants

DIFFUSE = ROOT / "diffuse"
OUT_DIR = ROOT / "data" / "diffuse"

MODULES = ["weighted", "smoothing", "scaling", "averaging", "discount"]

MODULE_TESTS = {
    "weighted": ["T_wmean_fail1", "T_wsum_fail", "T_wmean_pass1", "T_wmean_pass2", "T_wmean_pass3"],
    "smoothing": ["T_smooth_fail", "T_alpha_fail", "T_smooth_pass1", "T_smooth_pass2"],
    "scaling": ["T_scale_rows_fail", "T_scale_total_fail", "T_scale_pass1", "T_scale_pass2"],
    "averaging": ["T_average_fail", "T_window_fail", "T_average_pass1", "T_average_pass2"],
    "discount": ["T_discount_fail", "T_future_fail", "T_discount_pass1", "T_discount_pass2"],
}

MAX_MUTANTS = 60


def find_fault_mutant(clean_source: str) -> dict:
    """The seeded fault: the ``* 0`` literal changed to ``1``."""
    for mutant in generate_mutants(clean_source):
        if mutant["mutation_type"] == "LCR" and mutant["operator"] == "0" and mutant["detail"] == "1":
            return mutant
    raise RuntimeError("No LCR 0->1 fault mutant found")


def build(module: str) -> tuple[list[dict], list[dict]]:
    from diffuse import tests as oracle

    clean_source = (DIFFUSE / f"{module}.py").read_text()
    fault = find_fault_mutant(clean_source)
    buggy_source = fault["source"]
    fault_line = fault["line"]

    selected = [t for t in oracle.TESTS if t[0] in MODULE_TESTS[module]]
    baseline, coverage = baseline_results(buggy_source, selected)
    if all(result == "pass" for result in baseline.values()):
        raise RuntimeError(f"Fault in {module} does not fail any test")

    version_name = f"diffuse-{module}"
    rows: list[dict] = []
    for index, mutant in enumerate(generate_mutants(buggy_source)):
        results = mutant_results(
            mutant["source"], selected, module_name=f"__diffuse_{module}_{index}__"
        )
        metadata = node_metadata(
            buggy_source, mutant["line"], mutant["mutation_type"], mutant["detail"]
        )
        for test_id, _test_fn in selected:
            baseline_result = baseline[test_id]
            mutant_result = results[test_id]
            rows.append(
                {
                    "source_version": version_name,
                    "mutant_id": f"{version_name}::{module}::{mutant['mutant_id']}",
                    "line_number": mutant["line"],
                    "file": f"{module}.py",
                    "test_id": test_id,
                    "mutation_type": mutant["mutation_type"],
                    "mutated_operator": mutant["operator"],
                    "replacement_operator": mutant["detail"],
                    "ast_depth": metadata["ast_depth"],
                    "parent_node_type": metadata["parent_node_type"],
                    "test_covers_mutation": int(mutant["line"] in set(coverage[test_id])),
                    "baseline_test_result": baseline_result,
                    "outcome_changed": int(mutant_result != baseline_result),
                }
            )
    faults = [{"source_version": version_name, "fault_line": fault_line}]
    return rows, faults


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    all_faults: list[dict] = []
    for module in MODULES:
        rows, faults = build(module)
        all_rows.extend(rows)
        all_faults.extend(faults)
        print(
            f"  diffuse-{module}: {len(MODULE_TESTS[module])} tests, "
            f"fault line {faults[0]['fault_line']}, {len(rows)} pair rows"
        )
    pd.DataFrame(all_rows).to_csv(OUT_DIR / "pairs.csv", index=False)
    pd.DataFrame(all_faults).to_csv(OUT_DIR / "fault_labels.csv", index=False)
    print(f"Wrote {len(all_rows)} pair rows across {len(MODULES)} diffuse modules.")
    print(f"Pairs:  {OUT_DIR / 'pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())