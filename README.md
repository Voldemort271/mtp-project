# ML-PMT — Predictive Mutation-Based Fault Localization

A fault-localization pipeline for Python that **predicts** mutation-testing outcomes with a lightweight ML model instead of executing every mutant, delivering a behavioral (mutation-based) signal that beats real MBFL at near-spectrum-based (SBFL) cost.

It is evaluated on real BugsInPy bugs. Current slice: **32 bugs** across `tqdm`, `thefuck`, and `black` (with their fixed versions as clean training groups), ~25,500 test–mutant pairs, 296 known fault lines. The mutation engine covers operator swaps, literals, negation, argument removal, and index perturbation. The build is deterministic (sandboxed mutant runs, fixed hash seed, flaky-cell guard) and parallelized.

---

## Commands

Everything is run through the streamlined entry point (`scripts/mlpmt.py`), which delegates to focused scripts kept as the source of truth.

```bash
# install deps (once)
uv sync                      # or: uv pip install -r requirements.txt

# pipeline
uv run python scripts/mlpmt.py build        # build the BugsInPy historical corpus (add --workers 8 to parallelize)
uv run python scripts/mlpmt.py status       # show corpus/model status
uv run python scripts/mlpmt.py train        # fit + save model -> artifacts/model.joblib
uv run python scripts/mlpmt.py evaluate     # leave-one-bug-out Top-k / MRR / EXAM
uv run python scripts/mlpmt.py pairs        # pair-prediction metrics (AUC / AP / Brier)
uv run python scripts/mlpmt.py baselines    # compare vs SBFL and real-execution MBFL
uv run python scripts/mlpmt.py compare      # model-configuration comparison
uv run python scripts/mlpmt.py demo         # synthetic smoke demo

# line scoring can be the fixed Metallaxis formula (default) or a learned
# two-stage ranker (pair model + line ranker), which improves MRR on the slice:
uv run python scripts/mlpmt.py train --scoring two-stage
uv run python scripts/mlpmt.py evaluate --scoring two-stage

# rank lines for a NEW buggy program with a saved model
uv run python scripts/mlpmt.py localize \
  --candidates data/candidate_pairs.csv \
  --baseline-tests data/baseline_tests.csv

# tests
uv run python scripts/mlpmt.py test         # add `-- -q` to pass pytest args
```

The fine-grained scripts are also runnable directly: `scripts/build_bugs.py`, `scripts/evaluate_localization.py`, `scripts/evaluate_baselines.py`, `scripts/evaluate_diffuse.py`, `scripts/compare_models.py`, `scripts/benchmark_time.py`, `scripts/run_all.py`, `scripts/train_model.py`, `scripts/localize.py`.

---

## Reproducing from a fresh clone

The source, deps, and lockfile are committed; the BugsInPy benchmark environment is not (it is large and regenerable). One command provisions everything — project deps, the BugsInPy framework clone, per-project test venvs (Python 3.11), all buggy/fixed checkouts — and is idempotent:

```bash
uv run python scripts/setup_benchmark.py          # provision the environment
uv run python scripts/setup_benchmark.py --build  # ... and build the corpus afterwards
```

Notes:
- Requires `git`, `uv`, and network access (the first run clones BugsInPy and each project's history; checkouts take ~1–2 hours).
- `scripts/setup_benchmark.py` handles the per-project Python version and test deps (nose/mock for thefuck, attrs/typed-ast for black, etc.) and applies the thefuck conftest patch automatically via `build_bugs.py`.

Then reproduce the results:

```bash
uv run python scripts/build_bugs.py --workers 8   # ~45 min, deterministic
uv run python scripts/run_all.py                  # train + evaluate + baselines + timing
uv run python scripts/evaluate_diffuse.py         # diffuse-coverage benchmark (no benchmark env needed)
```

If you only want the parts that need no benchmark setup (engine, synthetic demo, diffuse benchmark, most tests), skip the provisioning and run `demo.py`, `build_corpus.py`, and `evaluate_diffuse.py` directly. The `data/pairs.csv`-based scripts and the leave-one-bug-out tests require the provisioned benchmark and corpus build to have run.

---

## Motivation: why predict mutation outcomes?

Fault localization ranks source lines by how likely they contain the bug. Two families dominate:

- **SBFL** (Ochiai, Tarantula): one coverage run; ranks lines by how much failing tests execute them. Cheap and strong when the failing test's executed lines distinguish the fault, but coverage is a weak signal when many lines share the same execution path.
- **MBFL** (Metallaxis): ranks lines whose *mutants change test outcomes* — a behavioral signal. It is more accurate than SBFL **when a rich mutation operator set can express the fault** (a mutant on the fault line flips a failing test), but that advantage is expensive: every mutant must be executed against the test suite.

The results on this slice qualify that "MBFL is more accurate" claim: even with our operator set (MOR, ROOR, COR, ASR, LCR, NOTR, AAR, INX, TSM, FCS), many faults have no outcome-flipping mutant, so real MBFL (MRR 0.41) does **not** beat SBFL (0.83). The lesson is not that MBFL is weak, but that the mutation signal only pays off where the operators can express the fault. That motivates this architecture:

- **ML-PMT** learns to *predict* mutation outcomes, so the behavioral signal can be had at SBFL-like cost instead of paying per-mutant execution.
- **The hybrid** fuses the predicted-mutation signal onto SBFL's coverage universe. It does not beat SBFL on MRR, but it has the best mean-rank and Top-10 — a developer inspects fewer positions to find a fault — and it matches ML-PMT's low cost.

The premise remains the standard MBFL assumption — lines whose synthetic mutants perturb test outcomes are more likely to hold the real fault — with the executions moved from inference into a one-time, amortized training step.

---

## Architecture

Four stages, wired end to end:

| Stage | Module | What it does |
|---|---|---|
| Mutation engine | `mutation_engine/` | LibCST transformers generate mutants: **MOR** (`+ - * / @`), **ROOR** (`== != < <= > >=`, `in`, `is`), **COR** (`and`/`or`), **ASR** (`+= -= *= /=`), **LCR** (numeric/boolean literals), **NOTR** (`not` insert/remove), **AAR** (argument removal), **INX** (index/slice perturbation), **TSM** (reshape args), **FCS** (call swaps). Operators are attributed to the operator token's line so multi-line boolean expressions localize correctly. |
| Execution runner | `execution/` | Runs real pytest per mutant in a subprocess; per-test results from junitxml and per-test line coverage from pytest-cov contexts. (`runner.py` is the in-process runner for the synthetic demo.) |
| Feature extraction | `features/` | One row per `(source_version, mutant, test)` with `mutation_type`, `mutated_operator`, `replacement_operator`, AST depth, parent node type, per-test coverage, and baseline test result. |
| Model & ranker | `ml_pmt/` | Gradient-boosted classifier (ordinal-encoded operator features) predicts per-test outcome-change probabilities, aggregated into an Ochiai/Metallaxis-style line score. |

**Ranking formula** (Metallaxis adaptation):

$$Suspiciousness(s) = \frac{f_{killed}(s)}{\sqrt{F \times (f_{killed}(s) + p_{killed}(s))}}$$

where `F` is the number of failing tests, and `f_killed`/`p_killed` are the predicted failing- and passing-test counts whose outcomes change for at least one mutant on line `s` (multiple mutants per line collapse by maximum probability per test).

---

## Pipeline vs MBFL-real

Both methods use the same mutants and the same ranking formula; they differ in where execution happens.

```
TRAINING (one-time, amortized):
  for each historical bug:
    generate mutants → execute each against the tests → label per-test outcome changes
  ── this IS MBFL-real execution, done once and reused as training labels ──

INFERENCE (per new bug):
  ML-PMT:    2 test runs (1 baseline for pass/fail + 1 coverage)
             → generate mutants (no execution)
             → predict per-test outcome-change probabilities
             → rank lines
  MBFL-real: 1 baseline test run
             → execute EVERY mutant against the tests (~54 test-suite runs/bug here)
             → rank lines from real outcomes
  SBFL:      1 coverage run → rank by coverage overlap
```

**Cost comparison** on this slice (test-suite runs per bug):

| Method | Execution cost per new bug | Where |
|---|---|---|
| SBFL | 1 coverage run | cheapest |
| ML-PMT (inference) | 2 test runs (baseline + coverage) + ~70 ms model | same order as SBFL |
| MBFL-real | ~54 mutant test-suite runs | paid every bug |

The corpus build (~47 min parallel for 32 bugs) is the amortized MBFL-real pass: train once, then ~0 marginal execution per bug. The build runs mutants in disposable sandboxes with a fixed hash seed and a flaky-cell guard, so a killed run cannot corrupt the source checkouts.

---

## Performance comparison

Measured on the thirty-two-bug slice with leave-one-bug-out (the bug's own buggy and fixed versions are excluded from training). The **hybrid** fuses the predicted mutation signal onto SBFL's full line universe: each line scores `max(normalized Ochiai, normalized mutation score)`. The `time_s` column is a coarse per-bug model (`executions × measured per-run time`); the exact wall-clock is in the benchmark below.

```
                         top1  top5  top10    mrr   mean-rank  exam  exam@10%  exam@20%  executions  time_s
SBFL-ochiai             0.750 0.906  0.938  0.834   4.281    0.579    0.125     0.312       1.0     1.73
SBFL-tarantula          0.750 0.906  0.938  0.834   4.281    0.579    0.125     0.312       1.0     1.73
Hybrid (SBFL+pred)      0.688 0.906  0.969  0.801   2.625    0.578    0.125     0.312       2.0     3.46
ML-PMT (two-stage)      0.562 0.844  0.969  0.689   2.906    0.255    0.344     0.500       2.0     3.46
ML-PMT (max formula)    0.344 0.781  0.844  0.536   4.406    0.340    0.219     0.438       2.0     3.46
MBFL-real (Metallaxis)  0.281 0.500  0.625  0.405  10.125    0.632    0.156     0.219      54.2   110.05
```

Pair-prediction model: ROC AUC ≈ 0.85, average precision ≈ 0.39.

**Exact wall-clock per bug** (`scripts/benchmark_time.py`, `time.perf_counter` around real work; MBFL's number is the serial sum of actually running every mutant's test suite):

```
Mean per bug: MBFL-real 106 s | ML-PMT 70 s | Hybrid 70 s | SBFL 4.6 s
```

Total wall-clock is only ~1.5× faster for ML-PMT because both methods pay the mutant **generation** cost (~12 s shared, and up to minutes on the largest `black` files) — a slow implementation, not a structural cost. The structural win is in **test execution**: MBFL runs the suite ~54×/bug, ML-PMT 2×. Excluding the shared generator, ML-PMT is ~2–10× faster on execution alone.

Inference cost: MBFL executes ~54 mutants/bug (each running the full suite); ML-PMT/Hybrid run the suite **2×** (baseline + coverage, no mutant execution); SBFL **1×** (coverage) — ML-PMT uses **~27× fewer test executions** than MBFL-real.

**Reading these honestly:**
- **SBFL remains strongest on MRR/Top-1** (0.83/0.75) — the coverage-heavy corpus favors it; the mutation signal alone (real or predicted) does not beat it.
- **The hybrid has the best mean-rank (2.63) and Top-10 (0.97)** at ML-PMT's cost; **ML-PMT two-stage has the best EXAM (0.26) and EXAM@20% (0.50)** — on the debugging-relevant metrics (positions inspected), the mutation-based methods are competitive or better than SBFL.
- **ML-PMT beats real MBFL on every accuracy metric** (two-stage MRR 0.69 vs 0.41) while using ~27× fewer test executions — the cost-savings claim holds.
- **Not a fair comparison to published BugsInPy numbers yet**: the slice is small (32 bugs), the engine still only expresses a subset of fault types, and EXAM/rank are comparable within the full-universe group (SBFL/Hybrid) and the mutant-universe group (ML-PMT/MBFL), not strictly across them.

### Where SBFL is weak: the diffuse-coverage benchmark

`diffuse/` + `scripts/evaluate_diffuse.py` provide a complementary benchmark in which every test covers the fault line (diffuse coverage), so coverage cannot discriminate the fault. There, the mutation signal dominates:

```
                              top1    mrr  rank   exam
ML-PMT (diffuse LOO)           1.0  1.000   1.0  0.102
MBFL-real                      0.8  0.840   1.8  0.169
SBFL-ochiai                    0.0  0.263   4.0  0.497
ML-PMT (trained on BugsInPy)   0.0  0.112   9.2  0.923
```

This demonstrates the complementarity: **SBFL wins when coverage is discriminative; the mutation signal (real, and predicted when trained on the same distribution) wins when coverage is diffuse.** It also shows the predicted model does not transfer across fault distributions without matching training data.

Caveat on the diffuse numbers: the modules were engineered so that, in most cases, only the fault line has a mutant that flips a failing test — so a perfect score there is partly an artifact of the construction. This benchmark shows the *mechanism* (mutation finds faults coverage cannot), not that the predicted model generalizes to arbitrary diffuse code.

---

## Project structure

```
target_model/        # synthetic NumPy Markowitz subject + oracle (smoke demo)
diffuse/             # diffuse-coverage benchmark modules (SBFL-weak setting)
mutation_engine/     # LibCST mutators (MOR, ROOR, COR, ASR, LCR, NOTR, AAR, INX, TSM, FCS) + generator
execution/           # subprocess pytest runner (+ coverage contexts); in-process runner
features/            # AST depth + parent-node extraction
ml_pmt/              # classifier, label derivation, grouped split, ranking, predict+rank
scripts/
├── mlpmt.py         # streamlined entry point (all modes)
├── build_bugs.py    # BugsInPy historical corpus builder
├── build_corpus.py  # synthetic corpus (smoke tests)
├── scan_bugs.py     # diagnostic: fault lines + mutability per bug
├── compare_models.py# model-configuration comparison
├── evaluate_baselines.py   # SBFL + real-MBFL comparison (full metrics table)
├── benchmark_time.py       # exact wall-clock timing per method
├── build_diffuse.py        # diffuse-coverage benchmark corpus
├── evaluate_diffuse.py     # SBFL vs MBFL/ML-PMT on the diffuse set
├── run_all.py              # full train→evaluate→benchmark pipeline
├── evaluate_localization.py# leave-one-bug-out metrics
├── evaluate_model.py       # pair metrics
├── train_model.py / localize.py / demo.py / run_tests.py
tests/               # test_model, test_scripts, test_upstream (19 tests)
data/                # generated pairs.csv + fault_labels.csv
benchmarks/          # BugsInPy framework + per-bug buggy/fixed checkouts + venvs
docs/ML-PMT.md       # full project spec
pyproject.toml       # deps + pytest config
requirements.txt
```

Full design rationale and implementation details: [docs/ML-PMT.md](docs/ML-PMT.md).