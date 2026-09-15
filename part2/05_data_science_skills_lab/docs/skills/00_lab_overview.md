# Lab overview

The Data Science Skills Mastery Lab is a teaching workbench. It pairs a browsable catalog of
analytical techniques with five workspaces that each run a real analysis on real public data.

## The five workspaces

| Workspace | Data | Question | Method |
|---|---|---|---|
| Classification | Titanic passenger list (OpenML 40945, 1,309 rows) | Who survived? | Gradient-boosted trees, stratified hold-out |
| Regression | Ames, Iowa house sales (OpenML 42165, 1,460 rows) | What will this house sell for? | Random forest on a log1p target |
| Imbalanced | ULB credit-card fraud (OpenML 1597, 284,807 rows, 0.172% fraud) | Which transactions are fraud? | Logistic regression, unweighted vs class-weighted |
| Analytics | Online Retail II (UCI, ~1.07M lines) | Are customers coming back, and did the variant win? | Cohort retention, weekly decomposition, two-proportion z-test |
| Quality | Online Retail II slice | Can we trust this table? | Rate-based quality scorecard |

## Running it

```bash
conda activate cmpe255
pip install -r requirements.txt
python scripts/prepare_data.py     # one-time download into data/cache/
pytest -q
streamlit run app/main.py
```

## What this lab does not do

Being explicit about the boundaries matters as much as the demonstrations:

- Single hold-out splits throughout. There is no cross-validation.
- No hyperparameter search. Parameters are pinned in `skills_lab/config.py` and reported.
- No resampling (no SMOTE) and no synthetic minority generation.
- No forecasting. The revenue decomposition is descriptive.
- One illustrative element: the checkout funnel, labelled as such wherever it appears,
  because Online Retail II records completed orders rather than page-level events.
