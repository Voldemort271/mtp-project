## PROJECT_SPEC.md: Predictive Mutation-Based Fault Localization (ML-PMT)

## 1. Project Overview

This project implements a Predictive Mutation-Based Fault Localization (ML-PMT) pipeline designed specifically for mathematical and Machine Learning architectures. Traditional mutation testing is too computationally expensive for ML pipelines because it requires full test suite execution for every generated mutant.

This architecture aims to reduce repeated mutant execution by using a lightweight machine learning classifier to predict whether each test's pass/fail outcome changes for a given mutant. The predicted test–mutant outcomes are aggregated into a mutation-based suspiciousness score for each line, allowing rapid debugging of silent logical errors (e.g., matrix shape mismatches) in models like Physics-Informed Neural Networks (PINNs) or Markowitz Mean-Variance Optimization scripts.

- Target Subject & Oracle: A deterministic Python mathematical pipeline (e.g., a Markowitz portfolio optimizer) validated by a strict pytest suite containing assertions for mathematical constraints (e.g., weights.sum() == 1.0, matrix dimensions).
- AST Mutation Engine: A custom syntax transformer built on LibCST that parses the target Python source code and systematically injects domain-specific bugs without losing formatting.
- Ground Truth Execution Runner: A controller that executes the project's tests against each mutant and records a pass/fail outcome-change label for each test–mutant pair.
- Feature Extraction Pipeline: An analyzer that extracts syntax and execution metadata (mutation operator type, AST nesting depth, parent node context, per-test coverage, and the original test result).
- Predictive Classifier & Ranker: A RandomForestClassifier that predicts per-test probabilities of an outcome change for each mutant. These predictions are aggregated and passed to the line-level Ochiai-style formula below.

The ranking quality rests on an explicit empirical premise: lines whose synthetic mutants are predicted to perturb test outcomes are more likely to contain the real fault. This is the standard mutation-based fault-localization assumption; it must be validated on faults with known locations, not assumed.

## 2. System Architecture & Core Modules

The pipeline has four stages:

1. **Mutation Engine**: A LibCST-based syntax transformer generates candidate mutants and keeps a ledger of `(mutant_id, original_line_number, mutation_type)`.
2. **Execution Runner**: Runs the target's tests against each mutant and records per-test outcomes plus per-test line coverage, not just the suite exit code.
3. **Feature Extraction**: Builds one row per `(source_version, mutant, test)` with structural and execution metadata.
4. **Predictive Model & Ranker**: A RandomForestClassifier estimates per-test outcome-change probabilities, which are aggregated into the line-level suspiciousness score below.

All four stages are implemented: stages 1 and 2 in `mutation_engine/` and `execution/`, stages 3 and 4 in `features/` and `ml_pmt/`. Two corpus builders exist: `scripts/build_corpus.py` produces a synthetic NumPy corpus for smoke tests, and `scripts/build_bugs.py` builds the primary **historical corpus from BugsInPy** real bugs. `scripts/evaluate_localization.py` measures stage 4's ranking quality.

## 3. Implementation Phases

Each phase must be fully tested before the next begins. Phases 1 and 2 produce the training data; Phases 3 and 4 consume it.

### Phase 1: The LibCST Mutation Engine

Implemented in `mutation_engine/` (`mutators.py` for candidate discovery, `generator.py` for mutant generation). `cst.CSTTransformer` traverses the concrete syntax tree and applies general-purpose mutation operators:

- Math Operator Replacement (MOR): Swap `+`, `-`, `*`, `/`, `@` in arithmetic expressions.
- Relational Operator Replacement (ROOR): Swap `==`, `!=`, `<`, `<=`, `>`, `>=`, and membership/identity `in`, `not in`, `is`, `is not`.
- Conditional Operator Replacement (COR): Swap `and` and `or`.
- Tensor Shape Mutation (TSM): Swap the two positional arguments of a reshape call, e.g. `.reshape(batch, -1)` to `.reshape(-1, batch)`.
- Function-Call Swapping (FCS): Swap equivalent library calls symmetrically, e.g. `np.sum` ↔ `np.prod` and `np.linalg.inv` ↔ `np.linalg.pinv`.

The earlier activation-swap operator (AOR, `nn.ReLU` ↔ `nn.Sigmoid`) applies only to PyTorch-based subjects; it is omitted. Mutants that serialize back to the original source, or duplicate an existing mutant, are dropped. Each mutant is recorded with `mutant_id`, `line`, `mutation_type`, `detail`, and `function`.

### Phase 2: Ground Truth Generation

Two runners implement this phase, depending on the subject:

- `execution/runner.py` loads single synthetic modules in-process and runs tests under `coverage.py`, giving per-test results and per-test covered lines.
- `execution/pytest_runner.py` runs **real multi-file projects** by invoking pytest in a subprocess, capturing per-test results from junitxml and per-test coverage via pytest-cov coverage contexts. This is the runner used for the BugsInPy historical corpus.

For this proposal, a test–mutant pair is labeled 1 when that test's pass/fail outcome changes relative to the unmutated (possibly faulty) program, and 0 otherwise. Collection errors, import errors, timeouts, and flaky results are infrastructure/invalid-mutant outcomes, not ordinary labels; both runners treat any test exception as a failure.

Semantics note: this "outcome change" label is a deliberate redefinition of the classic mutation-testing "killed" status. Standard Metallaxis counts a failing test as killing a mutant when the test still fails on the mutant (fail→fail). Here, fail→fail is labeled 0 because the mutation did not perturb the observable outcome; only flips (fail→pass, pass→fail) are labeled 1. A mutation on the true fault line that merely keeps a failing test failing therefore contributes nothing to that line's score. This is coherent but behaves differently from published MBFL and must be compared empirically against the standard kill definition.

For each test–mutant pair, generate a feature vector containing:

- `mutation_type`: One-hot encoded operator (MOR, ROOR, COR, TSM, FCS, etc.).
- `mutated_operator`: the exact symbol/call being mutated (`+`, `<`, `and`, `np.sum`, `reshape`).
- `replacement_operator`: what the mutant replaces it with (the mutant's `detail`).
- `ast_depth` and `parent_node_type`: Structural context of the mutated node.
- `test_covers_mutation`: Whether this test executes the original mutation location.
- `baseline_test_result`: Whether this test passes or fails on the unmutated program.
- Optional test features: assertion/oracle category and stable coverage summaries. The feature contract forbids raw test IDs or test-unique identifiers, because they would let the model memorize labels and bypass the source-version grouping.

Train a RandomForestClassifier on historical test–mutant pairs, predicting the probability that the pair's outcome changes. Split evaluation data by source version or project, not by randomly splitting related mutants. At inference, run the unmutated program once to obtain test results and coverage, generate candidate mutants, and predict pair probabilities without executing those mutants.

Calculate the suspiciousness score for each line $s$ using the Ochiai-style formula:

$$Suspiciousness(s) = \frac{f_{killed}(s)}{\sqrt{F \times (f_{killed}(s) + p_{killed}(s))}}$$

(Where $F$ is the number of failing tests on the unmutated program, and $f_{killed}(s)$ and $p_{killed}(s)$ are the predicted failing- and passing-test counts whose outcomes change for at least one mutant on line $s$. For version 1, collapse multiple mutants on a line to one probability per test using the maximum predicted pair probability, $q_{s,t}=\max_{m\in M_s}q_{m,t}$; then sum $q_{s,t}$ over failing and passing tests. This avoids counting a test repeatedly because multiple operators target the same line. The formula is the Metallaxis adaptation of Ochiai; the "killed" counts are the outcome-change expectations defined above. Passing-test flips (pass→fail) appear only in the denominator and therefore reduce a line's score; this matches published Metallaxis semantics and is an explicit modeling choice.)

### Phase 3: Feature Engineering

Implemented across `features/extractor.py` and the corpus builders. `node_metadata` computes `ast_depth` (ancestor count) and `parent_node_type` for a mutated node from the CST; the corpus builders combine these with per-test coverage (`test_covers_mutation`) and baseline results into one row per `(source_version, mutant, test)`:

- `ml_pmt/features.py` derives the `outcome_changed` label by comparing `baseline_test_result` with the mutant's test result, and provides `grouped_train_test_split` to hold out whole source versions.
- The feature contract is defined in `FEATURE_COLUMNS`: `mutation_type`, `ast_depth`, `parent_node_type`, `test_covers_mutation`, `baseline_test_result`, `mutated_operator`, `replacement_operator`. The last two are categorical and let the model distinguish the "repair" mutant from other mutants on the same line — without them, every mutant on a line shares an identical feature vector and the repair cannot be singled out.

`scripts/build_corpus.py` seeds eight synthetic faulty variants plus a clean base program and writes `data/pairs.csv` and `data/fault_labels.csv` (smoke-test data). The primary historical corpus comes from `scripts/build_bugs.py`, described below.

### Phase 4: Model Training & Fault Localization

`ml_pmt/classifier.py` fits the outcome-change model on pair rows. The default is a `HistGradientBoostingClassifier` with **ordinal encoding** of the categorical features and inverse-class-frequency sample weights. Two model choices measurably improved leave-one-bug-out ranking on the seven-bug slice:

- **Ordinal instead of one-hot encoding** of `mutated_operator`/`replacement_operator`: one-hot produces an all-zero vector for unseen operator pairs (common in leave-one-bug-out), while ordinal lets each tree split on the operators independently. MRR 0.55 → 0.59.
- **Gradient boosting instead of RandomForest**: MRR 0.59 → 0.60 with the worst case (tqdm-2) improving from rank 12 to 4. `scripts/compare_models.py` reproduces this comparison across configs.

Because most pairs are labeled 0 (no outcome change), expect heavy class imbalance; `scripts/evaluate_model.py` reports average precision, ROC AUC, and Brier score, with average precision preferred for ranking signal. Brier score is not the optimization target here — the ranking uses summed probabilities, so calibration trades off against the ordinal-encoding ranking gains.

`ml_pmt/localization.py` collapses predicted probabilities per `(line, test)` using the maximum, then applies the suspiciousness formula; `ml_pmt/pipeline.py` combines prediction and ranking. `scripts/localize.py` ranks lines from a saved model and a candidate CSV.

Two line-scoring architectures are available (`--scoring max` or `--scoring two-stage`):

- **max** (default): the hand-fixed Metallaxis formula on per-test maximum probabilities.
- **two-stage** (`ml_pmt/ranker.py`): keeps the pair model, collapses its predictions into per-line aggregate features (failing/passing mass, mutant count, coverage, probability extremes), and *learns* the suspiciousness score with a small classifier trained on known fault lines.

Important: the two-stage ranker must be evaluated with **nested leave-one-bug-out** — each training bug's line features must come from a pair model that has not seen that bug. A naive evaluation that trains the line ranker on in-sample pair predictions inflates the result badly (MRR 0.89 vs the honest 0.74 on this slice), because the ranker is trained on unrealistically clean features. Under honest evaluation the two-stage ranker (MRR 0.74) is on par with the fixed max formula (MRR 0.77), not better, so the fixed formula remains the default. `scripts/score_experiments.py` reproduces the aggregation experiments (max/mean/failing-only/weighted/two-stage).

Fault-localization accuracy is measured by `scripts/evaluate_localization.py` using leave-one-bug-out (excluding both the buggy and fixed versions of the held-out bug) and ranking metrics: Top-1/Top-5 accuracy, MRR, and EXAM score. Remaining work is comparing against spectrum-based (SBFL) and conventional mutation-based baselines, and against the standard (non-outcome-change) kill definition.

## Historical Corpus: BugsInPy Slice

`scripts/build_bugs.py` builds the primary training corpus from real BugsInPy bugs. The slice currently spans eight bugs across two low-dependency projects, each included with its fixed version as a clean training group (16 source versions, ~5100 test–mutant pairs, 29 known fault lines):

| Version | Target failing tests | Fault pattern | Operators at fault |
|---|---|---|---|
| tqdm-2 | `test_format_meter` | missing `ncols` guard in `disp_trim`/format output | none (control flow) |
| tqdm-5 | `test_bool` | `total` inferred too late when disabled | `and` (COR) |
| tqdm-9 | `test_si_format`, `test_update` | rounding thresholds + `__len__` on `None` | `<` (ROOR) |
| thefuck-5 | `test_match_bitbucket` | branch-name handling | `and`, `in` |
| thefuck-6 | `test_get_new_command` | quoted branch names | `and` (COR) |
| thefuck-7 | `test_match` | `php -s` detection | `in` (ROOR) |
| thefuck-30 | `test_not_file` | file-existence condition | `and` (COR) |
| thefuck-32 | `test_match` | `ls -lah` detection | `and` (COR) |

For each bug, `build_bugs.py` parses the buggy-vs-fixed diff to obtain the known fault lines, derives the failing tests from `run_test.sh`, curates a small passing-test set from the same file, generates operator mutants of the fault files (capped, fault-line mutants prioritized), and runs real pytest per mutant via `execution/pytest_runner.py`. Old test suites are made runnable on modern pytest by applying a small conftest patch for thefuck (`prepare_checkout`) and using Python 3.11 (nose/mock compatibility). Fault lines come from every changed line of the fix, so a version is "found" if any of its fault lines is ranked.

Run it with:

```bash
uv run python scripts/build_bugs.py
```

This overwrites `data/pairs.csv` and `data/fault_labels.csv` with the historical corpus. The build is **deterministic**: mutants run in disposable sandbox copies (the source checkouts are never touched), with a fixed hash seed, isolated `HOME`, sandbox state cleaned between mutants, and each mutant's tests run twice with unstable cells dropped. It is **parallelized** across bugs and versions (`--workers`, default 1; 0 = auto): the eight-bug slice builds in ~5 minutes with 8 workers instead of ~20 serial.

### Results on the historical slice

```bash
uv run python scripts/evaluate_localization.py
```

With the enriched features, gradient-boosting model, and the eight-bug slice, under honest (nested) leave-one-bug-out: Top-1 ≈ 0.62, Top-5 ≈ 0.88, MRR ≈ 0.77, EXAM ≈ 0.59 with the fixed max formula; the two-stage ranker is on par (Top-1 0.62, MRR 0.74). Six of eight bugs rank the fault first with max; tqdm-2 and tqdm-9 are found within the top 3. Adding the `in`/`is` operators plus the deterministic build roughly doubled the pair-model signal (positive rate 0.10 → 0.28, AP 0.36 → 0.67).

### Baselines

```bash
uv run python scripts/evaluate_baselines.py
```

Runs three comparison methods on the same slice — real-execution MBFL (Metallaxis with the executed `outcome_changed` labels), and SBFL (Ochiai and Tarantula from coverage vectors) — alongside the predicted model:

```
method                     top1  top5   mrr  exam
Hybrid (SBFL+pred)        0.667 0.833 0.765 0.459
SBFL-ochiai               0.583 0.833 0.712 0.498
SBFL-tarantula            0.583 0.833 0.712 0.498
ML-PMT (max formula)      0.500 0.667 0.621 0.620
ML-PMT two-stage          0.417 0.833 0.601 0.602
MBFL-real (Metallaxis)    0.500 0.667 0.617 0.721
```

**The decisive finding:** on a three-project slice, mutation-only localization (real *and* predicted) does not beat SBFL — mutation ranking only works where a mutant on the fault line flips a failing test, and it ranks only lines that have mutants, while SBFL ranks every covered line. The fix is the **hybrid**: fuse the predicted mutation signal onto SBFL's full line universe (`max` of normalized Ochiai and normalized mutation score). This beats SBFL (MRR 0.77 vs 0.71) and is never worse per bug, because non-mutant lines keep their coverage score and the prediction only boosts. This reframes the architecture's contribution: the mutation-prediction signal is a *booster for spectrum-based localization*, not a standalone MBFL replacement, and it keeps the ~1-test-run inference cost. SBFL's EXAM is lower only because it shares the same (full-universe) denominator here.

## 4. Worked Example: From a Faulty Program to a Ranked Line

This example fixes the prediction target and shows exactly how pair predictions become a line score. The classifier predicts test–mutant outcome changes, not an aggregate whole-suite mutant-survival label.

### Program and baseline tests

Assume the source contains this deliberate fault on line 3:

```python
import numpy as np

def expected_return(weights, returns):
    return float(np.sum(weights + returns))  # line 3: faulty; should be a dot product

def normalize_weights(weights):
    return weights / np.sum(weights)  # line 6
```

Run four deterministic tests against the unmutated, faulty source:

| Test | Operation and expected result | Baseline result |
|---|---|---|
| T1 | `expected_return(np.array([.25, .75]), np.array([.10, .20]))` is approximately `.175` | Fail (faulty result: `1.3`) |
| T2 | `expected_return(np.array([1, 0]), np.array([0, -1]))` is approximately `0` | Pass (result: `0`) |
| T3 | `normalize_weights(np.array([2, 2]))` is approximately `[.5, .5]` | Pass |
| T4 | `expected_return(np.array([.5, .5]), np.array([.15, .15]))` is approximately `.15` | Fail (faulty result: `1.3`) |

Thus the failing-test set is $F_s = \{T1,T4\}$ and the passing-test set is $P_s = \{T2,T3\}$, with $F=2$.

### Mutants and ground-truth pair labels

Generate one candidate mutant per line for this small example:

- **M3** (line 3): replace `+` with `*`. This repairs the faulty calculation for T1 and T4.
- **M6** (line 6): replace `/` with `*`.

Define the pair label $K_{m,t}=1$ exactly when test $t$ changes pass/fail status between the unmutated program and mutant $m$. In the offline ground-truth runner, run each test against both versions and record this comparison:

| Mutant | T1 (fail) | T2 (pass) | T3 (pass) | T4 (fail) |
|---|---:|---:|---:|---:|
| M3, line 3 | 1 (fail → pass) | 0 (pass → pass) | 0 (line not covered) | 1 (fail → pass) |
| M6, line 6 | 0 (line not covered) | 0 (line not covered) | 1 (pass → fail) | 0 (line not covered) |

Each training row is a `(source version, mutant, test)` tuple. For example, an M3/T1 row has features such as `mutation_type=MOR_MULTIPLY`, the mutated node's AST depth and parent type, `test_covers_mutation=1`, and `baseline_test_result=fail`; its ground-truth label is 1. These labels are collected by executing mutants in the training corpus. The held-out inference example below does not use the table's ground-truth labels.

### Inference and suspiciousness calculation

Suppose a classifier trained on other source versions predicts the following probabilities for this program. These values are illustrative model outputs, not measured results:

| Candidate line / mutant | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|
| Line 3 / M3 | 0.90 | 0.15 | 0.02 | 0.85 |
| Line 6 / M6 | 0.02 | 0.02 | 0.80 | 0.02 |

For line 3, the predicted failing-test mass is $f_{killed}(3)=0.90+0.85=1.75$; the passing-test mass is $p_{killed}(3)=0.15+0.02=0.17$. Therefore:

$$Suspiciousness(3)=\frac{1.75}{\sqrt{2\times(1.75+0.17)}}\approx0.893$$

For line 6, $f_{killed}(6)=0.02+0.02=0.04$ and $p_{killed}(6)=0.02+0.80=0.82$, giving:

$$Suspiciousness(6)=\frac{0.04}{\sqrt{2\times(0.04+0.82)}}\approx0.030$$

The ranking is therefore **line 3, then line 6**, which puts the deliberately faulty line first. If several mutants target a line, version 1 uses the maximum pair probability for each `(line, test)` before applying the formula. Evaluate other aggregation policies separately rather than changing the score definition between runs.

## 5. Model API & Scripts

The ranking and prediction core is in `ml_pmt/`:

- `features.py`: derives outcome-change labels from baseline and mutant test results, and provides a source-version-grouped train/test split.
- `classifier.py`: one-hot encodes categorical features and fits a RandomForestClassifier.
- `localization.py`: applies the documented maximum-per-line/test aggregation and suspiciousness formula.
- `pipeline.py`: predicts candidate pair probabilities and returns the ranked lines without executing the candidate mutants.

Training data must contain one row per `(source_version, mutant, test)` with the feature columns above plus `mutant_test_result`. A basic training flow is:

```python
import pandas as pd
from ml_pmt import (
    OutcomeChangeModel,
    add_outcome_change_labels,
    grouped_train_test_split,
    predict_and_rank,
)

pairs = add_outcome_change_labels(pd.read_csv("data/pairs.csv"))
train_rows, held_out_rows = grouped_train_test_split(pairs)
model = OutcomeChangeModel().fit(train_rows)

# candidate_pairs contains line_number, test_id, and the five model features.
ranking = predict_and_rank(
    model,
    candidate_pairs,
    baseline_test_results={"T1": "fail", "T2": "pass"},
)
```

The primary corpus in `data/` is the **BugsInPy historical slice** (see above); `scripts/build_corpus.py` can regenerate a small synthetic corpus for smoke tests. The probabilities in the worked example are illustrative. The tests verify model execution and score calculations, and `evaluate_localization.py` measures ranking quality on real fault lines. None of this is evidence of predictive accuracy on unseen real projects beyond the measured slice.

### Convenience scripts

`scripts/mlpmt.py` is the streamlined entry point — one command per mode, with sensible defaults on the corpus in `data/`:

```bash
uv run python scripts/mlpmt.py build        # build the BugsInPy historical corpus
uv run python scripts/mlpmt.py status       # show corpus/model status
uv run python scripts/mlpmt.py train        # fit and save model -> artifacts/model.joblib
uv run python scripts/mlpmt.py evaluate     # leave-one-bug-out Top-k/MRR/EXAM
uv run python scripts/mlpmt.py pairs        # pair metrics (AUC/AP/Brier)
uv run python scripts/mlpmt.py baselines    # vs SBFL and real-execution MBFL
uv run python scripts/mlpmt.py compare      # model-configuration comparison
uv run python scripts/mlpmt.py demo         # synthetic smoke demo
uv run python scripts/mlpmt.py test         # run the project test suite (use `--` for pytest args)
```

The fine-grained scripts it delegates to are kept as the source of truth, so each mode also works standalone:

```bash
uv run python scripts/build_bugs.py
uv run python scripts/evaluate_localization.py
uv run python scripts/evaluate_baselines.py
uv run python scripts/evaluate_model.py --pairs data/pairs.csv
uv run python scripts/compare_models.py
uv run python scripts/train_model.py --pairs data/pairs.csv --output artifacts/model.joblib
uv run python scripts/localize.py \
  --model artifacts/model.joblib \
  --candidates data/candidate_pairs.csv \
  --baseline-tests data/baseline_tests.csv \
  --output artifacts/ranked_lines.csv
```

`build_bugs.py` is the end-to-end historical upstream stage: it checks out real BugsInPy bugs, derives fault lines from the fix diffs, runs real pytest per mutant, and writes `data/pairs.csv` and `data/fault_labels.csv`. `evaluate_localization.py` then trains leave-one-bug-out and reports ranking metrics on the known fault lines. The training CSV needs `source_version`, the five model features, and either `outcome_changed` or `mutant_test_result` (the latter is compared with `baseline_test_result`). The candidate CSV needs `line_number`, `test_id`, and the model features. The baseline-test CSV has `test_id,result`, with `result` set to `pass` or `fail`. `demo.py` builds a tiny synthetic dataset to smoke-test training and ranking.

## 6. Proposed File Structure

Implemented layout:

```
target_model/
├── model.py        # NumPy Markowitz subject (synthetic demo only)
├── test_cases.py   # deterministic mathematical oracle (8 tests)
mutation_engine/
├── mutators.py     # LibCST candidate discovery (MOR, ROOR, COR, TSM, FCS)
└── generator.py    # mutant generation with ledger
execution/
├── runner.py       # in-process per-test results + coverage (synthetic)
└── pytest_runner.py# subprocess pytest runner + per-test coverage contexts (real projects)
features/
└── extractor.py    # AST depth + parent-node extraction
ml_pmt/
├── __init__.py
├── classifier.py   # RandomForest outcome-change model
├── features.py     # outcome-change labels + grouped split
├── localization.py # Ochiai-style suspiciousness ranking
└── pipeline.py     # predict + rank composition
scripts/
├── mlpmt.py              # streamlined entry point (build/train/evaluate/pairs/baselines/compare/localize/demo/test/status)
├── run_tests.py            # run pytest from the repository root
├── build_bugs.py           # BugsInPy historical corpus
├── build_corpus.py         # synthetic corpus (smoke tests)
├── scan_bugs.py            # diagnostic: fault lines + mutability per bug
├── compare_models.py       # leave-one-bug-out comparison of model configs
├── score_experiments.py    # scoring/aggregation experiments (max/mean/two-stage)
├── evaluate_baselines.py   # SBFL + real-MBFL comparison
├── evaluate_localization.py# leave-one-bug-out ranking metrics
├── evaluate_model.py       # held-out source-version pair metrics
├── train_model.py          # fit and save a model from pair CSV
├── localize.py             # rank lines from a saved model
└── demo.py                 # synthetic train/predict/rank smoke demo
tests/
├── test_model.py
├── test_scripts.py
└── test_upstream.py
data/               # generated pairs.csv and fault_labels.csv
benchmarks/
├── BugsInPy/       # benchmark framework and per-bug metadata
└── work/           # per-bug buggy/fixed checkouts + per-project venvs
docs/
└── ML-PMT.md
pyproject.toml      # project deps + pytest config (testpaths confined to tests/)
requirements.txt
```

## 7. Agent Action Plan (Execution Instructions)

To the AI Coding Agent: Proceed step-by-step. Do not move to the next step until the current one is fully tested and verified.

- 1. Step 1: Initialize the project structure and create a mock model.py (a basic Mean-Variance Optimization pipeline using numpy) and test_model.py to serve as our target. Ensure the tests pass. **Done**: `target_model/model.py` and `target_model/test_cases.py`.
- 2. Step 2: Implement mutation_engine/mutators.py. Create a LibCST transformer that successfully targets and modifies mathematical operators and matrix/tensor shape calls. Write a test to ensure the CST serializes back to valid Python code. **Done**: `mutation_engine/`.
- 3. Step 3: Implement the generator.py and runner.py scripts. The generator must create a ledger (mutants.json) mapping mutants to modified lines. The runner must execute the actual test suite for each mutant and append per-test outcome-change labels. **Done**: `mutation_engine/generator.py`, `execution/runner.py` (in-process), and `execution/pytest_runner.py` (subprocess, for real projects).
- 4. Step 4: Build extractor.py to parse the original code, determine AST depth and per-test coverage, and format test–mutant feature rows into a Pandas DataFrame. **Done**: `features/extractor.py`, `scripts/build_corpus.py` (synthetic), and `scripts/build_bugs.py` (historical BugsInPy corpus).
- 5. Step 5: Implement the RandomForestClassifier in train.py. Train it on the generated DataFrame. **Done**: `ml_pmt/classifier.py`, `scripts/train_model.py`.
- 6. Step 6: Implement localize.py. Use a faulty model with a known line, run the prediction pipeline without executing candidate mutants, apply the Ochiai-style formula, and report the ranked list of suspicious lines. **Done**: `ml_pmt/localization.py`, `scripts/localize.py`, plus `scripts/evaluate_localization.py` for leave-one-out ranking metrics.