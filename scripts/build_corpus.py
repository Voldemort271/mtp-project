"""Build the bundled synthetic training corpus and fault labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from execution.runner import baseline_results, mutant_results
from features.extractor import node_metadata
from mutation_engine.generator import generate_mutants
from target_model import test_cases

BASE_SOURCE = (ROOT / "target_model" / "model.py").read_text()

VARIANT_SPECS = [
    ("v_er_mul_add", "expected_return", "MOR", "+"),
    ("v_er_mul_sub", "expected_return", "MOR", "-"),
    ("v_er_sum_prod", "expected_return", "FCS", "np.prod"),
    ("v_pv_at_add", "portfolio_variance", "MOR", "+"),
    ("v_nw_sum_prod", "normalize_weights", "FCS", "np.prod"),
    ("v_nw_div_mul", "normalize_weights", "MOR", "*"),
    ("v_mw_at_mul", "markowitz_weights", "MOR", "*"),
    ("v_tr_reshape_swap", "transpose_returns", "ARC", "swap"),
]


def build_corpus(
    variant_ids: list[str] | None = None,
    *,
    include_clean: bool = True,
    out_dir: Path = ROOT / "data",
) -> dict:
    """Generate pair rows for the requested faulty variants plus the clean base."""
    rows: list[dict] = []
    fault_rows: list[dict] = []

    base_mutants = generate_mutants(BASE_SOURCE)
    specs = {name: spec for name, *spec in VARIANT_SPECS}
    names = variant_ids if variant_ids is not None else list(specs)

    for variant_id in names:
        if variant_id not in specs:
            raise ValueError(f"Unknown variant {variant_id!r}")
        function, mutation_type, detail = specs[variant_id]
        matched = _mutant_for(base_mutants, function, mutation_type, detail)
        if matched is None:
            raise ValueError(f"No base mutant for {variant_id}")
        _append_variant(
            rows,
            variant_id,
            matched["source"],
            fault_line=matched["line"],
            fault_rows=fault_rows,
        )

    if include_clean:
        _append_variant(rows, "v_clean", BASE_SOURCE, fault_line=None, fault_rows=None)

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.DataFrame(rows)
    pairs_path = out_dir / "pairs.csv"
    pairs.to_csv(pairs_path, index=False)

    fault_frame = pd.DataFrame(fault_rows)
    faults_path = out_dir / "fault_labels.csv"
    fault_frame.to_csv(faults_path, index=False)

    return {
        "pairs_path": str(pairs_path),
        "faults_path": str(faults_path),
        "num_rows": len(pairs),
        "num_variants": len(pairs["source_version"].unique()) if len(pairs) else 0,
        "num_faults": len(fault_frame),
    }


def _mutant_for(
    base_mutants: list[dict],
    function: str,
    mutation_type: str,
    detail: str | None,
) -> dict | None:
    for mutant in base_mutants:
        if (
            mutant["function"] == function
            and mutant["mutation_type"] == mutation_type
            and mutant["detail"] == detail
        ):
            return mutant
    return None


def _append_variant(
    rows: list[dict],
    variant_id: str,
    variant_source: str,
    *,
    fault_line: int | None,
    fault_rows: list[dict] | None,
) -> None:
    baseline, coverage = baseline_results(variant_source, test_cases.TESTS)
    if fault_rows is not None:
        fault_rows.append({"source_version": variant_id, "fault_line": fault_line})

    mutants = generate_mutants(variant_source)
    for index, mutant in enumerate(mutants):
        results = mutant_results(
            mutant["source"],
            test_cases.TESTS,
            module_name=f"__mlpmt_{variant_id}_{index}__",
        )
        metadata = node_metadata(
            variant_source,
            mutant["line"],
            mutant["mutation_type"],
            mutant["detail"],
        )
        for test_id, _test_fn in test_cases.TESTS:
            baseline_result = baseline[test_id]
            mutant_result = results[test_id]
            rows.append(
                {
                    "source_version": variant_id,
                    "mutant_id": f"{variant_id}::{mutant['mutant_id']}",
                    "line_number": mutant["line"],
                    "test_id": test_id,
                    "mutation_type": mutant["mutation_type"],
                    "mutated_operator": mutant["operator"],
                    "replacement_operator": mutant["detail"],
                    "ast_depth": metadata["ast_depth"],
                    "parent_node_type": metadata["parent_node_type"],
                    "test_covers_mutation": int(
                        mutant["line"] in set(coverage[test_id])
                    ),
                    "baseline_test_result": baseline_result,
                    "outcome_changed": int(
                        mutant_result != baseline_result
                    ),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data",
        help="directory to write pairs.csv and fault_labels.csv",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="exclude the clean (correct) base program from the corpus",
    )
    args = parser.parse_args()

    summary = build_corpus(out_dir=args.out, include_clean=not args.no_clean)
    print(
        f"Wrote {summary['num_rows']} pair rows across "
        f"{summary['num_variants']} source versions and "
        f"{summary['num_faults']} known faults."
    )
    print(f"Pairs:  {summary['pairs_path']}")
    print(f"Faults: {summary['faults_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())