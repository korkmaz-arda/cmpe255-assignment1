# Data Science Skills Mastery Lab

An interactive teaching workbench: a browsable catalog of analytical techniques, each paired
with a workspace that **actually runs** the analysis it describes, on **real public data**.

The lab's subject is correct methodology rather than leaderboard performance. Four ideas recur
in every workspace: split before you fit, choose a metric that matches the decision, set
decision thresholds deliberately on data you are not reporting, and keep statistical
significance separate from practical significance.

Nothing on screen is hardcoded. If a number is displayed, it was computed on your machine from
the data in `data/cache/`; if a workspace cannot be computed, the app says so instead of
showing a plausible stand-in.

## Quick start

```bash
conda activate cmpe255
pip install -r requirements.txt

python scripts/prepare_data.py      # one-time download (~200 MB, a few minutes)
pytest -q                           # 49 tests, offline, ~5s
streamlit run app/main.py           # opens on http://localhost:8501

python -m skills_lab.status         # service status / dataset readiness
```

No training step is needed: the five benchmarks are fitted at app startup in under two
seconds and cached for the session. There are no saved model artifacts - reproducibility comes
from pinned seeds and a pinned configuration in `skills_lab/config.py`.

## Data

All four datasets are public and require no credentials. `scripts/prepare_data.py` is the only
code that touches the network; it normalises each source into `data/cache/*.parquet`, which is
gitignored and never committed.

| Dataset | Source | Size | Used by |
|---|---|---|---|
| Titanic passenger list | OpenML data id 40945 | 1,309 rows | Classification |
| Ames, Iowa house sales | OpenML data id 42165 | 1,460 rows | Regression |
| Credit-card fraud (ULB) | OpenML data id 1597 | 284,807 rows, 492 frauds (0.172%) | Imbalanced classification |
| Online Retail II | UCI archive | 1,067,371 lines | Cohorts, revenue, quality audit |

The fraud table is used in full rather than subsampled: logistic regression on 29 features is
fast, and sampling would either distort the base rate or leave too few positives to evaluate.

One element of the app is not real data: the five-stage checkout funnel. Online Retail II
records completed orders, not page-level events, and no comparable free event stream exists,
so those stage counts are a teaching example and are labelled
"illustrative - not measured from an event stream" wherever they appear.

## The five workspaces

**Classification - Titanic survival.** Gradient-boosted trees (100 stages, depth 3, learning
rate 0.08) on a stratified 75/25 hold-out. Preprocessing is fitted on the training partition
only, inside a pipeline. `boat` and `body` are excluded as post-outcome leakage. Includes a
**live predictor**: the controls are scored by the fitted pipeline itself, with `family_size`
and `is_alone` recomputed exactly as in training.

**Regression - Ames house prices.** Random forest (100 trees, max depth 10) fitted on
`log1p(SalePrice)`, back-transformed with `expm1` before scoring, so RMSE, MAE and R-squared
are all in dollars.

**Imbalanced - credit-card fraud.** Two logistic regressions differing only in `class_weight`,
on a stratified 55/15/30 train/validation/test split. Average precision is the primary metric;
ROC-AUC is secondary context. The F1-optimal decision threshold is chosen from **every**
threshold on the validation precision-recall curve - not a fixed 0.1-0.9 grid, which cannot
reach the optimum at a 0.17% base rate - and the selected rule is then applied **once** to the
test partition and reported separately. The interactive workbench measures on validation, so
exploring it cannot contaminate that final report. Thresholds are expressed on the decision
function (log-odds), because class weighting saturates predicted probabilities at 1.0.

**Business analytics - Online Retail II.** Monthly cohort retention over every observable
period; daily revenue over the last 60 days with recorded transactions, decomposed additively
with a 7-day period; and a two-proportion A/B
calculator. Both ambiguous definitions are stated on screen: revenue is net of returns (with
gross sales alongside), and the retention population is customers with a non-null id whose
invoice is a product line, is not a credit note, and has positive quantity and price.

**Data quality - Online Retail II slice.** Completeness, validity, uniqueness and consistency
as normalised 0-100 rates, combined with documented weights (0.35 / 0.30 / 0.20 / 0.15) into
the composite score. Health states come from score and severity thresholds, never from a raw
issue count. Returns, credit notes, adjustment codes and statistical outliers are reported as
**business events** and **review flags**, not as defects.

## A/B testing behaviour

The calculator runs on counts you enter - the project ships no stored experiment record and
no example numbers, so the fields start empty and nothing is computed until you fill them in.
It reports the pooled two-proportion z statistic, a two-tailed normal p-value, absolute and
relative lift, and a 95% confidence interval on the difference. The recommendation weighs
effect direction, effect magnitude against a practical threshold you set, and the width of the
interval; it is not `p < 0.05` alone. Post-hoc power is shown as descriptive context and is
explicitly excluded from the decision.

## Layout

```
skills_lab/           importable analytical core - no Streamlit imports anywhere
  config.py           seeds, split fractions, hyperparameters, weights, thresholds
  data/               one module per source + acquire.py (the only network code)
  benchmarks/         the five computations
  catalog.py          the skills catalog and its view routing
  stats.py            two-proportion test, CI, descriptive power
  status.py           service status (also `python -m skills_lab.status`)
app/                  Streamlit layer: shell, caching, views, modals, glossary
docs/skills/          seven standalone methodology guides
scripts/              prepare_data.py
tests/                offline, fixture-driven
```

## Measured results

Recomputed on every run; these were the values at the time of writing.

| Workspace | Result |
|---|---|
| Classification | ROC-AUC 0.888, F1 0.765, accuracy 82.9% on 328 held-out passengers |
| Regression | R-squared 0.892, RMSE $27,485, MAE $17,988 (median sale price $163,000) |
| Imbalanced | Average precision 0.720 weighted vs 0.710 unweighted; recall at the default cutoff 87% vs 61%; validation-selected cutoff gives test F1 0.803 against validation F1 0.845 |
| Quality | Composite 98.7/100 (PASS); completeness 96.9%; 211 duplicate rows in 20,000 |
| Analytics | 25 monthly cohorts; 60-day revenue series |

## What this lab deliberately does not do

- No cross-validation: single hold-out splits throughout.
- No hyperparameter search. Parameters are pinned and reported, not tuned.
- No resampling (no SMOTE), no focal loss, no polynomial expansion, no target encoding.
- No forecasting. The seasonal decomposition is descriptive.
- No model persistence, no database, no user accounts.

## Tests

```bash
pytest -q          # everything
pytest -q -k leak  # the train-only preprocessing invariants
```

Tests are offline and fixture-driven, so they run without the parquet cache. Tests that need
the real data (`test_real_data_smoke.py`, `test_app_smoke.py`) skip themselves with a message
telling you to run `scripts/prepare_data.py`. The suite covers the leakage invariant, the
log-target round trip, threshold selection reading validation and never test, PR-curve-derived
candidate cutoffs, the z-test against `statsmodels`, quality scoring and its scale invariance,
business events versus defects, cohort triangularity and determinism, live-predictor wiring,
total skill routing, and that every catalog jump target reaches a real workspace.
