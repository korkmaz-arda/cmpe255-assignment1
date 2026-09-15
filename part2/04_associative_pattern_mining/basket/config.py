"""Paths, dataset coordinates, thresholds and validation ranges.

Every tunable that materially affects behaviour lives here so the pipeline, the
search, the tests and the dashboard all read the same numbers.

This module is deliberately free of any UI-framework import: the whole ``basket``
package must be importable and testable with Streamlit absent.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
#
# The environment overrides exist so tests can redirect every write to tmp_path
# without monkeypatching each module that happens to touch disk.
# --------------------------------------------------------------------------- #

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent


def _dir(env_var: str, default: Path) -> Path:
    return Path(os.environ.get(env_var, default))


DATA_DIR = _dir("BASKET_DATA_DIR", PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = _dir("BASKET_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")

RAW_ARCHIVE = RAW_DIR / "instacart.zip"
BASKETS_PARQUET = PROCESSED_DIR / "baskets.parquet"      # one row per sampled order
CATALOG_PARQUET = PROCESSED_DIR / "catalog.parquet"      # one row per catalog product
CORPUS_META_JSON = PROCESSED_DIR / "corpus_meta.json"    # measured facts about the split

RULES_JSON = ARTIFACT_DIR / "rules.json"
GRAPH_JSON = ARTIFACT_DIR / "graph.json"
BENCHMARKS_JSON = ARTIFACT_DIR / "benchmarks.json"
CATALOG_JSON = ARTIFACT_DIR / "catalog.json"
RUN_META_JSON = ARTIFACT_DIR / "run_meta.json"
AUTORESEARCH_JSON = ARTIFACT_DIR / "autoresearch.json"

# --------------------------------------------------------------------------- #
# Dataset
#
# The Instacart Online Grocery Basket Analysis data, via a public Kaggle mirror
# that serves without credentials. Only four of the six archive members are used;
# the 32M-row `prior` split and `orders.csv` are not needed for basket mining.
# --------------------------------------------------------------------------- #

DATASET_NAME = "Instacart Online Grocery Basket Analysis"
DATASET_SLUG = "psparks/instacart-market-basket-analysis"
DATASET_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
DATASET_PAGE = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"

ARCHIVE_MEMBERS = (
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
)

# --------------------------------------------------------------------------- #
# Corpus construction
# --------------------------------------------------------------------------- #

SEED = 42

# A product enters the mining catalog when it appears in at least this share of
# orders, measured over the FULL train split before any sampling, so the catalog
# is a property of the dataset rather than of whichever orders we happened to
# draw. The floor is 5x the production support threshold: an itemset can never be
# more frequent than its rarest item, so a product below this line has no
# headroom to take part in a frequent pair. On the train split this admits 256
# products. It is a frequency rule, applied blind to what the products are.
CATALOG_MIN_FREQUENCY = 0.005

# Orders in the mining corpus. The default is the entire train split — the corpus
# is the whole available population, not a sample of it. Smaller values draw a
# seeded, content-blind random sample (F10's corpus-size control).
TRAIN_SPLIT_ORDERS = 131_209
DEFAULT_N_ORDERS = TRAIN_SPLIT_ORDERS

# The parameter search runs on the same full corpus as the production pass, so
# its mean-lift numbers are directly comparable to the rule set the app serves.
SEARCH_N_ORDERS = TRAIN_SPLIT_ORDERS

# Illustrative prices. Instacart ships no prices; these are generated from a
# seeded RNG keyed on product_id, live in the catalog only, and are never read by
# mining, rule scoring, graph construction or recommendation ranking.
PRICE_SEED = 1042
PRICE_MIN = 0.99
PRICE_MAX = 14.99

# --------------------------------------------------------------------------- #
# Mining thresholds
#
# Measured against the real support distribution of this corpus (see README);
# they are not carried over from the source project's synthetic 33-product
# corpus, where supports were an order of magnitude larger.
# --------------------------------------------------------------------------- #

MAX_LEN = 4                      # benchmark + production passes

# Real grocery baskets at product granularity are far sparser than the source
# project's synthetic 33-product corpus, so these floors are an order of
# magnitude lower than the ones in the specification. They are set by evidence
# count, not by the rule counts they produce: on the 131,209-order train split a
# support of 0.001 means an itemset is backed by ~131 real orders, and 0.0015 by
# ~197. The benchmark pass uses the higher support with looser rule gates, the
# production pass the lower support with stricter ones, mirroring the structure
# of the specification's threshold table.
BENCHMARK = {
    "min_support": 0.0015,
    "max_len": MAX_LEN,
    "min_confidence": 0.20,
    "min_lift": 1.20,
}

PRODUCTION = {
    "min_support": 0.0010,
    "max_len": MAX_LEN,
    "min_confidence": 0.25,
    "min_lift": 1.25,
}

# Search baseline mirrors the benchmark pass, as in the specification.
SEARCH_BASELINE = dict(BENCHMARK)

# Phase 4 optimises 3-4 item antecedents. A 4-item antecedent needs an itemset of
# at least length 5, so that phase alone lifts the cap, and lowers the support
# floor far enough that a 5-product combination can clear it at all. On this
# corpus that takes roughly 40 supporting orders — thin evidence, which the
# phase's reflection states rather than glosses over.
BUNDLE_MAX_LEN = 5
BUNDLE_MIN_SUPPORT = 0.0003

# --------------------------------------------------------------------------- #
# Search acceptance rule
# --------------------------------------------------------------------------- #

MIN_LIFT_GAIN = 0.02             # a trial must beat the incumbent mean lift by more than this
PHASE2_MIN_RULES = 10            # ... and keep at least this many rules
PHASE3_MIN_RULES = 8

# --------------------------------------------------------------------------- #
# Serving / UI
# --------------------------------------------------------------------------- #

GRAPH_TOP_RULES = 50             # strongest pairwise rules projected into the network
GRAPH_VISIBLE_EDGES = 35         # edges drawn when no product is selected
GRAPH_RADIUS = 1.0
RECOMMENDATION_COUNT = 6
UPLIFT_TOP_N = 3                 # suggestions summed into the illustrative add-on value
RULES_TABLE_FEED = 60            # rules fetched for the explorer
RULES_TABLE_ROWS = 50            # rows displayed

# Re-mining control ranges (F10). All four affect the regenerated rule set.
SUPPORT_RANGE = (0.0005, 0.0100)
CONFIDENCE_RANGE = (0.10, 0.90)
LIFT_RANGE = (1.0, 5.0)
ORDERS_RANGE = (5_000, TRAIN_SPLIT_ORDERS)  # upper bound = every order in the train split

# Numerical floor for the conviction denominator, so confidence == 1 yields a
# large finite number rather than a division by zero.
EPSILON = 1e-9

FALLBACK_COLOR = "#8892a4"

# Instacart ships 21 departments. Colours are assigned deterministically by
# department_id so a department keeps its colour across re-mines.
DEPARTMENT_PALETTE = (
    "#4c9be8", "#f2994a", "#27ae60", "#eb5757", "#9b51e0",
    "#2d9cdb", "#f2c94c", "#56ccf2", "#bb6bd9", "#6fcf97",
    "#e07a5f", "#81b29a", "#c9ada7", "#6d6875", "#457b9d",
    "#e63946", "#a8dadc", "#ffb4a2", "#b5838d", "#7f8c8d",
    "#3d5a80", "#98c1d9", "#ee6c4d", "#293241",
)


def department_color(department_id: int) -> str:
    """Stable colour for a department (F05's one-colour-per-department contract)."""
    return DEPARTMENT_PALETTE[int(department_id) % len(DEPARTMENT_PALETTE)]
