"""Central configuration: paths, seeds, split fractions, model hyperparameters.

Everything that makes a run reproducible lives here so the values are visible in one
place rather than scattered through the benchmark modules.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"

SEED = 42

# --- Public data sources -------------------------------------------------------------
# OpenML dataset ids. Downloaded by scripts/prepare_data.py, never committed.
OPENML_TITANIC = 40945          # Titanic passenger list (~1309 rows)
OPENML_AMES = 42165             # Ames, Iowa house sale prices (1460 rows)
OPENML_CREDITCARD = 1597        # ULB credit-card fraud (284,807 rows, 0.172% positive)
ONLINE_RETAIL_II_URL = (
    "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
)

CACHE_FILES = {
    "titanic": CACHE_DIR / "titanic.parquet",
    "ames": CACHE_DIR / "ames.parquet",
    "creditcard": CACHE_DIR / "creditcard.parquet",
    "retail": CACHE_DIR / "online_retail_ii.parquet",
}

# --- Split fractions -----------------------------------------------------------------
CLASSIFICATION_TEST_SIZE = 0.25   # stratified
REGRESSION_TEST_SIZE = 0.25
FRAUD_TEST_SIZE = 0.30            # stratified; remainder split into train/validation
FRAUD_VALIDATION_SIZE = 0.15      # of the whole dataset -> 55 / 15 / 30

# --- Model hyperparameters -----------------------------------------------------------
GB_PARAMS = dict(n_estimators=100, max_depth=3, learning_rate=0.08, random_state=SEED)
RF_PARAMS = dict(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
LOGREG_PARAMS = dict(max_iter=1000, random_state=SEED)

# --- Analysis parameters -------------------------------------------------------------
ALPHA = 0.05                      # two-sided significance level for the A/B test
REVENUE_SERIES_DAYS = 60          # trailing window for the daily revenue series
SEASONAL_PERIOD = 7               # weekly seasonality for the decomposition
QUALITY_SAMPLE_ROWS = 20_000      # deterministic head slice profiled by the quality audit

# Quality composite weights (documented in the UI and README; must sum to 1.0).
QUALITY_WEIGHTS = {
    "completeness": 0.35,
    "validity": 0.30,
    "uniqueness": 0.20,
    "consistency": 0.15,
}

# Score -> health state thresholds. Used for the dataset score and every column score.
HEALTH_PASS_MIN = 90.0
HEALTH_WARNING_MIN = 70.0
