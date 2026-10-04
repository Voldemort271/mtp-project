import pandas as pd
import pytest

from ml_pmt import (
    OutcomeChangeModel,
    add_outcome_change_labels,
    grouped_train_test_split,
    predict_and_rank,
    rank_suspicious_lines,
)


def test_add_outcome_change_labels_marks_only_pass_fail_flips():
    pairs = pd.DataFrame(
        {
            "baseline_test_result": ["fail", "pass", "pass", "fail"],
            "mutant_test_result": ["pass", "pass", "fail", "fail"],
        }
    )

    labeled = add_outcome_change_labels(pairs)

    assert labeled["outcome_changed"].tolist() == [1, 0, 1, 0]
    assert pairs.columns.tolist() == ["baseline_test_result", "mutant_test_result"]


def test_add_outcome_change_labels_rejects_non_test_results():
    pairs = pd.DataFrame(
        {
            "baseline_test_result": ["pass"],
            "mutant_test_result": ["error"],
        }
    )

    with pytest.raises(ValueError, match="only 'pass' or 'fail'"):
        add_outcome_change_labels(pairs)


def test_grouped_split_keeps_source_versions_disjoint():
    pairs = _training_pairs()

    train, test = grouped_train_test_split(pairs, test_size=0.25)

    assert set(train["source_version"]).isdisjoint(test["source_version"])
    assert len(train) + len(test) == len(pairs)


def test_model_fits_and_predicts_probabilities_for_unknown_categories():
    training = _training_pairs()
    model = OutcomeChangeModel(n_estimators=30, random_state=7).fit(training)
    inference = pd.DataFrame(
        [
            {
                "mutation_type": "MOR",
                "mutated_operator": "+",
                "replacement_operator": "-",
                "ast_depth": 3,
                "parent_node_type": "If",
                "test_covers_mutation": 1,
                "baseline_test_result": "fail",
            },
            {
                "mutation_type": "ROOR",
                "mutated_operator": "<",
                "replacement_operator": "<=",
                "ast_depth": 2,
                "parent_node_type": "Return",
                "test_covers_mutation": 0,
                "baseline_test_result": "pass",
            },
        ]
    )

    probabilities = model.predict_probabilities(inference)

    assert probabilities.shape == (2,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_predict_and_rank_runs_the_full_inference_path():
    model = OutcomeChangeModel(n_estimators=30, random_state=7).fit(
        _training_pairs()
    )
    candidate_pairs = pd.DataFrame(
        [
            {
                "line_number": line,
                "test_id": test_id,
                "mutation_type": mutation,
                "mutated_operator": operator,
                "replacement_operator": replacement,
                "ast_depth": depth,
                "parent_node_type": parent,
                "test_covers_mutation": covered,
                "baseline_test_result": result,
            }
            for line, mutation, operator, replacement, depth, parent, covered in [
                (3, "MOR", "*", "+", 4, "BinaryOperation", 1),
                (6, "MOR", "/", "*", 2, "Return", 1),
            ]
            for test_id, result in [
                ("T1", "fail"),
                ("T2", "pass"),
                ("T3", "pass"),
                ("T4", "fail"),
            ]
        ]
    )

    ranked = predict_and_rank(
        model,
        candidate_pairs,
        {"T1": "fail", "T2": "pass", "T3": "pass", "T4": "fail"},
    )

    assert ranked["line_number"].tolist() in ([3, 6], [6, 3])
    assert ranked["suspiciousness"].between(0, 1).all()


def test_ranker_matches_worked_example_and_uses_max_per_line_test():
    predictions = pd.DataFrame(
        [
            # Line 3 / M3
            (3, "T1", 0.90),
            (3, "T2", 0.15),
            (3, "T3", 0.02),
            (3, "T4", 0.85),
            # A second line-3 mutant cannot cause T1 to be counted twice.
            (3, "T1", 0.70),
            # Line 6 / M6
            (6, "T1", 0.02),
            (6, "T2", 0.02),
            (6, "T3", 0.80),
            (6, "T4", 0.02),
        ],
        columns=["line_number", "test_id", "probability"],
    )

    ranked = rank_suspicious_lines(
        predictions,
        {"T1": "fail", "T2": "pass", "T3": "pass", "T4": "fail"},
    )

    assert ranked["line_number"].tolist() == [3, 6]
    assert ranked.loc[0, "f_killed"] == pytest.approx(1.75)
    assert ranked.loc[0, "p_killed"] == pytest.approx(0.17)
    assert ranked.loc[0, "suspiciousness"] == pytest.approx(0.893043, abs=1e-6)
    assert ranked.loc[1, "f_killed"] == pytest.approx(0.04)
    assert ranked.loc[1, "p_killed"] == pytest.approx(0.82)
    assert ranked.loc[1, "suspiciousness"] == pytest.approx(0.0305, abs=1e-4)


def _training_pairs():
    rows = []
    examples = [
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "fail", 1),
        ("MOR", "*", "+", 4, "BinaryOperation", 1, "pass", 0),
        ("ROOR", "<", ">", 3, "Comparison", 1, "fail", 1),
        ("ROOR", "<", ">", 3, "Comparison", 1, "pass", 0),
        ("COR", "and", "or", 5, "BooleanOperation", 0, "fail", 0),
        ("COR", "and", "or", 5, "BooleanOperation", 0, "pass", 0),
        ("MOR", "/", "*", 2, "Return", 1, "pass", 1),
        ("MOR", "/", "*", 2, "Return", 1, "fail", 0),
    ]
    for version in range(4):
        for mutation, operator, replacement, depth, parent, covered, baseline, label in examples:
            rows.append(
                {
                    "source_version": f"project-{version}",
                    "mutation_type": mutation,
                    "mutated_operator": operator,
                    "replacement_operator": replacement,
                    "ast_depth": depth,
                    "parent_node_type": parent,
                    "test_covers_mutation": covered,
                    "baseline_test_result": baseline,
                    "outcome_changed": label,
                }
            )
    return pd.DataFrame(rows)
