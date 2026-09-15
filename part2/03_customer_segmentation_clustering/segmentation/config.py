"""Constants, paths, persona archetypes and validation ranges.

Every tunable that materially affects behaviour lives here so that the pipeline,
the tests and the dashboard all read the same numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

RAW_CSV = RAW_DIR / "marketing_campaign.csv"
CLEAN_PARQUET = PROCESSED_DIR / "customers.parquet"

# Artifact files written by the pipeline.
MODEL_BUNDLE = ARTIFACT_DIR / "model_bundle.joblib"      # model + scaler + column order
PCA_BUNDLE = ARTIFACT_DIR / "pca_bundle.joblib"          # fitted PCA (reused at inference)
PERSONAS_JSON = ARTIFACT_DIR / "personas.json"
BENCHMARKS_JSON = ARTIFACT_DIR / "benchmarks.json"       # leaderboard + production metrics
ELBOW_JSON = ARTIFACT_DIR / "elbow.json"
SCATTER_JSON = ARTIFACT_DIR / "scatter.json"
AUTORESEARCH_JSON = ARTIFACT_DIR / "autoresearch.json"
RUN_META_JSON = ARTIFACT_DIR / "run_meta.json"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

DATASET_NAME = "Customer Personality Analysis"
DATASET_SLUG = "imakash3011/customer-personality-analysis"
DATASET_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
DATASET_PAGE = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"
RAW_CSV_MEMBER = "marketing_campaign.csv"

# Customer enrolment dates in this release run through 2014; ages are measured
# against that reference year rather than "today" so the derivation is stable.
REFERENCE_YEAR = 2014

# --------------------------------------------------------------------------- #
# Seeds and sampling
# --------------------------------------------------------------------------- #

DEFAULT_SEED = 42
DEFAULT_K = 5
K_SWEEP = tuple(range(2, 10))            # F08: elbow / silhouette sweep over k
PROJECTION_SAMPLE = 1200                 # F06/S02: t-SNE + scatter sample size
TSNE_PERPLEXITY = 30
AUTORESEARCH_SAMPLE = 1600               # F10: subsample the hill climb operates on
AUTORESEARCH_GATE = 0.0005               # F10: silhouette improvement acceptance gate
# A candidate must also produce an actionable partition. Silhouette can be gamed
# by isolating a handful of outliers into singleton clusters — a 1580/7/1/11/1
# split scores well and segments nothing. Any cluster holding less than this
# share of the data disqualifies the candidate, applied uniformly to every trial.
MIN_CLUSTER_SHARE = 0.01
INFERENCE_EPSILON = 1e-5                 # F09: inverse-distance guard

# Retraining studio bounds (F11). Unlike the source, every one of these has effect.
MIN_K, MAX_K = 2, 10
MIN_TRAIN_SAMPLE = 500

# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #

BASE_FEATURES = [
    "age",
    "income_k",
    "spending_score",
    "recency_days",
    "total_spend",
    "web_visits_month",
    "discount_sensitivity",
    "household_size",
]

ENGINEERED_FEATURES = [
    "discretionary_ratio",
    "monetary_velocity",
    "digital_engagement",
    "deal_affinity",
]

FEATURE_COLUMNS = BASE_FEATURES + ENGINEERED_FEATURES

ATTRIBUTE_LABELS = {
    "age": "Age",
    "income_k": "Annual income (k$)",
    "spending_score": "Spending score (1–100)",
    "recency_days": "Recency (days)",
    "total_spend": "Total spend ($, 2-yr)",
    "web_visits_month": "Web visits / month",
    "discount_sensitivity": "Discount sensitivity (0–1)",
    "household_size": "Household size",
}

ATTRIBUTE_MEANINGS = {
    "age": "Customer age at the dataset's 2014 reference year.",
    "income_k": "Self-reported yearly household income, in thousands of dollars.",
    "spending_score": "Derived 1–100 purchase-engagement score (conversion efficiency + campaign responsiveness).",
    "recency_days": "Days since the customer's most recent purchase.",
    "total_spend": "Dollars spent across all six product categories over the two-year history.",
    "web_visits_month": "Visits to the company website in the last month.",
    "discount_sensitivity": "Share of purchases made with a discount.",
    "household_size": "Adults plus children and teenagers living in the home.",
}

# --------------------------------------------------------------------------- #
# Inference input validation (F09)
# --------------------------------------------------------------------------- #
# Bounds are calibrated to the cleaned dataset's observed domain, padded outward
# so plausible hypothetical customers are accepted while nonsense is rejected.

INPUT_RANGES: dict[str, tuple[float, float]] = {
    "age": (18.0, 90.0),
    "income_k": (1.0, 150.0),
    "spending_score": (1.0, 100.0),
    "recency_days": (0.0, 99.0),
    "total_spend": (0.0, 3000.0),
    "web_visits_month": (0.0, 20.0),
    "discount_sensitivity": (0.0, 1.0),
    "household_size": (1.0, 8.0),
}

INPUT_DEFAULTS: dict[str, float] = {
    "age": 45.0,
    "income_k": 52.0,
    "spending_score": 50.0,
    "recency_days": 49.0,
    "total_spend": 600.0,
    "web_visits_month": 5.0,
    "discount_sensitivity": 0.20,
    "household_size": 3.0,
}

# Fixed display maxima for the normalized persona radar (S11), chosen from the
# cleaned data's observed upper range so every axis uses a stable scale.
RADAR_AXES = ["income_k", "spending_score", "total_spend", "web_visits_month", "age", "recency_days"]
RADAR_MAXIMA = {
    "income_k": 105.0,
    "spending_score": 100.0,
    "total_spend": 2600.0,
    "web_visits_month": 12.0,
    "age": 75.0,
    "recency_days": 99.0,
}
RADAR_FLOOR = 0.05


# --------------------------------------------------------------------------- #
# Persona archetypes (F07)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Archetype:
    """A marketing persona identity plus the behavioural profile that defines it.

    ``profile`` holds the archetype's position on each attribute as a population
    percentile (0.5 = the median customer, 0.9 = higher than 90% of customers). Clusters are
    matched to archetypes by comparing their centroids on these axes, subject to
    the hard constraint that the cluster may not contradict the archetype on any
    ``signature`` axis.
    """

    key: str
    name: str
    tagline: str
    badge: str
    color: str
    description: str
    strategy: str
    profile: dict[str, float] = field(default_factory=dict)
    # The axes this persona's narrative makes explicit claims about. A cluster
    # that contradicts the archetype on any of them cannot receive it, no matter
    # how well the remaining axes agree: an averaged match score would otherwise
    # let six agreeing axes outvote the one the persona is actually named for.
    signature: tuple[str, ...] = ()


# Axes used for cluster <-> archetype matching. Deliberately the behavioural
# drivers of the personas, not every attribute.
# recency_days and household_size are included deliberately: archetype narratives
# describe how recently a customer last bought ("long gaps", "just purchased") and
# their household ("small households", "multi-person households"), and an axis
# that is not matched on is an axis whose claims nothing checks.
MATCH_AXES = ["income_k", "spending_score", "total_spend", "web_visits_month", "age",
              "discount_sensitivity", "recency_days", "household_size"]

# Signature axes are checked as directional CLAIMS, not as distances. Every
# position is the cluster mean's percentile within the whole customer base, so
# 0.5 is the median on every axis. An archetype's target on a signature axis says
# which claim its narrative makes:
#   target >= CLAIM_HIGH_TARGET  -> the narrative claims "high"
#   target <= CLAIM_LOW_TARGET   -> the narrative claims "low"
#   otherwise                    -> the narrative claims "about average"
# and the cluster must satisfy that claim. In plain terms:
#   "high"    = at or above the 60th percentile (10+ points past the median)
#   "low"     = at or below the 40th percentile
#   "average" = within 15 points of the median (35th to 65th percentile)
# The bands are calibrated against what the words mean to a reader, pinned by
# tests: a segment at the 30th percentile is not "mid-range", one at the 62nd is,
# and a $123 basket at the 34th percentile of spend is still "low".
# Checking |actual - target| instead would conflate how extreme a soft-matching
# target happens to be with whether the narrative's claim is actually true.
CLAIM_HIGH_TARGET = 0.65
CLAIM_LOW_TARGET = 0.35
CLAIM_HIGH_MIN = 0.60
CLAIM_LOW_MAX = 0.40
CLAIM_MID_BAND = (0.35, 0.65)

# Clusters that no catalogued persona fits without contradiction are described
# directly from their own data instead, using these colours.
UNMATCHED_COLORS = ["#64748B", "#78716C", "#6B7280", "#71717A", "#737373",
                    "#57534E", "#475569", "#52525B", "#44403C", "#334155"]

# Plain-language descriptors for a cluster's distinguishing traits: (low, high).
TRAIT_WORDS = {
    "income_k": ("lower income", "higher income"),
    "spending_score": ("low engagement", "high engagement"),
    "total_spend": ("low spend", "high spend"),
    "web_visits_month": ("light web use", "heavy web use"),
    "age": ("younger", "older"),
    "discount_sensitivity": ("rarely uses discounts", "discount-driven"),
    "recency_days": ("recent buyers", "long since last order"),
    "household_size": ("small households", "large households"),
}

ARCHETYPES: list[Archetype] = [
    Archetype(
        key="vip_champions",
        name="VIP Champions",
        tagline="High income, high engagement, high value",
        badge="👑",
        color="#7C3AED",
        description=(
            "Affluent customers who buy often, convert readily and respond to campaigns. "
            "They carry large baskets and rarely need a discount to act."
        ),
        strategy=(
            "Protect this revenue: early access to new ranges, a loyalty tier with concierge "
            "service, and premium bundles. Do not discount — it erodes margin they already pay."
        ),
        profile={"income_k": 0.90, "spending_score": 0.85, "total_spend": 0.90,
                 "web_visits_month": 0.25, "age": 0.60, "discount_sensitivity": 0.15, "recency_days": 0.4, "household_size": 0.35},
        signature=("income_k", "spending_score", "total_spend", "discount_sensitivity"),
    ),
    Archetype(
        key="prudent_affluents",
        name="Prudent Affluents",
        tagline="Comfortable means, careful spending",
        badge="🏦",
        color="#0EA5E9",
        description=(
            "Higher-income households that nonetheless buy sparingly and let long gaps open "
            "between orders. Capacity to spend is there; the motivation is not."
        ),
        strategy=(
            "Reactivation over acquisition: quality-and-durability messaging, curated "
            "replenishment reminders, and occasional high-signal offers rather than volume promotion."
        ),
        profile={"income_k": 0.85, "spending_score": 0.25, "total_spend": 0.35,
                 "web_visits_month": 0.45, "age": 0.65, "discount_sensitivity": 0.35, "recency_days": 0.7, "household_size": 0.45},
        signature=("income_k", "spending_score", "total_spend", "recency_days"),
    ),
    Archetype(
        key="young_trendsetters",
        name="Young Trendsetters",
        tagline="Modest budgets, heavy digital engagement",
        badge="🚀",
        color="#F59E0B",
        description=(
            "A younger segment with frequent site visits and strong campaign responsiveness on a "
            "limited budget. Discovery-driven rather than value-driven."
        ),
        strategy=(
            "Meet them where they browse: social and in-app campaigns, small-basket bundles, "
            "referral incentives and fast-moving seasonal drops."
        ),
        profile={"income_k": 0.30, "spending_score": 0.80, "total_spend": 0.30,
                 "web_visits_month": 0.90, "age": 0.15, "discount_sensitivity": 0.40, "recency_days": 0.45, "household_size": 0.2},
        signature=("age", "web_visits_month", "spending_score", "income_k"),
    ),
    Archetype(
        key="bargain_hunters",
        name="Bargain Hunters",
        tagline="Price-led, promotion-driven",
        badge="🏷️",
        color="#EF4444",
        description=(
            "Lower-income, low-basket customers who transact mainly when something is "
            "discounted, with a high share of their purchases made on a deal."
        ),
        strategy=(
            "Margin-aware promotion: clearance and end-of-line stock, threshold offers that "
            "lift basket size, and own-brand alternatives instead of blanket discounting."
        ),
        profile={"income_k": 0.15, "spending_score": 0.20, "total_spend": 0.10,
                 "web_visits_month": 0.75, "age": 0.45, "discount_sensitivity": 0.90, "recency_days": 0.5, "household_size": 0.6},
        signature=("income_k", "total_spend", "discount_sensitivity"),
    ),
    Archetype(
        key="mainstream_loyalists",
        name="Mainstream Loyalists",
        tagline="Mid-range income, engagement and spend",
        badge="🧭",
        color="#10B981",
        description=(
            "The dependable middle of the customer base: mid-range income, mid-range engagement "
            "and mid-sized baskets."
        ),
        strategy=(
            "Grow share of wallet: category cross-sell from their existing baskets, a simple "
            "points programme, and subscription options for repeat staples."
        ),
        profile={"income_k": 0.50, "spending_score": 0.50, "total_spend": 0.50,
                 "web_visits_month": 0.50, "age": 0.50, "discount_sensitivity": 0.50, "recency_days": 0.5, "household_size": 0.5},
        signature=("income_k", "spending_score", "total_spend"),
    ),
    # Spares — only used when the served k exceeds five (F11 allows k up to 10).
    Archetype(
        key="emerging_families",
        name="Emerging Families",
        tagline="Larger households, budget-conscious growth",
        badge="🏡",
        color="#6366F1",
        description=(
            "Multi-person households with sizeable but price-aware "
            "baskets — volume matters more to them than premium."
        ),
        strategy="Family-size packs, bulk thresholds and a back-to-season calendar.",
        profile={"income_k": 0.40, "spending_score": 0.45, "total_spend": 0.7,
                 "web_visits_month": 0.55, "age": 0.45, "discount_sensitivity": 0.65, "recency_days": 0.5, "household_size": 0.85},
        signature=("household_size", "total_spend", "discount_sensitivity"),
    ),
    Archetype(
        key="lapsing_browsers",
        name="Lapsing Browsers",
        tagline="Still looking, no longer buying",
        badge="💤",
        color="#94A3B8",
        description=(
            "Customers who still visit the site but whose last purchase is far behind them. "
            "Long gaps between orders, low conversion and low spend."
        ),
        strategy="Win-back sequence with a time-boxed incentive, then suppress if unconverted.",
        profile={"income_k": 0.45, "spending_score": 0.10, "total_spend": 0.15,
                 "web_visits_month": 0.85, "age": 0.50, "discount_sensitivity": 0.55, "recency_days": 0.85, "household_size": 0.55},
        signature=("recency_days", "spending_score", "total_spend"),
    ),
    Archetype(
        key="window_shoppers",
        name="Window Shoppers",
        tagline="Plenty of visits, almost no purchases",
        badge="🪟",
        color="#F472B6",
        description=(
            "Heavy browsers and very light buyers. They visit the site often but rarely convert, "
            "and their spending stays near zero."
        ),
        strategy=(
            "Remove the friction that stops the first real basket: a first-order incentive, "
            "clearer pricing and delivery terms, and abandoned-basket prompts. Cap the spend per "
            "head until one converts."
        ),
        profile={"income_k": 0.20, "spending_score": 0.05, "total_spend": 0.05,
                 "web_visits_month": 0.85, "age": 0.45, "discount_sensitivity": 0.7,
                 "recency_days": 0.50, "household_size": 0.55},
        signature=("web_visits_month", "spending_score", "total_spend"),
    ),
    Archetype(
        key="premium_occasionals",
        name="Premium Occasionals",
        tagline="Rare visits, large baskets",
        badge="💎",
        color="#DB2777",
        description=(
            "Rare but high-value shoppers: long gaps between orders, a light digital footprint, "
            "and an unusually large basket when they do buy."
        ),
        strategy="Event-led outreach — gifting seasons, anniversaries and high-ticket launches.",
        profile={"income_k": 0.75, "spending_score": 0.55, "total_spend": 0.80,
                 "web_visits_month": 0.15, "age": 0.60, "discount_sensitivity": 0.20, "recency_days": 0.8, "household_size": 0.4},
        signature=("recency_days", "total_spend", "web_visits_month"),
    ),
    Archetype(
        key="recent_high_rollers",
        name="Recent Big Spenders",
        tagline="Bought large, and bought just now",
        badge="⚡",
        color="#8B5CF6",
        description=(
            "Customers in the middle of an active buying spell: a large basket placed very "
            "recently, comfortable income and little reliance on discounts. Their defining trait "
            "is how fresh the purchase is, not how often they buy."
        ),
        strategy=(
            "Strike while the relationship is warm: post-purchase cross-sell on what they just "
            "bought, an accessory or replenishment prompt, and an invitation into the loyalty tier "
            "before the moment passes."
        ),
        profile={"income_k": 0.70, "spending_score": 0.70, "total_spend": 0.75,
                 "web_visits_month": 0.35, "age": 0.50, "discount_sensitivity": 0.20,
                 "recency_days": 0.03, "household_size": 0.4},
        signature=("recency_days", "total_spend", "income_k", "discount_sensitivity"),
    ),
    Archetype(
        key="value_regulars",
        name="Value Regulars",
        tagline="Highly engaged, mid-sized baskets",
        badge="🛒",
        color="#14B8A6",
        description=(
            "Engaged repeat buyers with mid-sized baskets who convert readily and respond to "
            "campaigns."
        ),
        strategy="Replenishment reminders, multi-buy offers and a low-friction loyalty card.",
        profile={"income_k": 0.35, "spending_score": 0.75, "total_spend": 0.5,
                 "web_visits_month": 0.65, "age": 0.40, "discount_sensitivity": 0.55, "recency_days": 0.35, "household_size": 0.55},
        signature=("spending_score", "total_spend"),
    ),
    Archetype(
        key="established_savers",
        name="Established Savers",
        tagline="Older, settled, deliberate",
        badge="🌿",
        color="#A16207",
        description=(
            "Older households with settled routines, low digital engagement and deliberate, "
            "infrequent purchasing."
        ),
        strategy="Catalogue and store-led communication; reliability and service messaging.",
        profile={"income_k": 0.55, "spending_score": 0.25, "total_spend": 0.5,
                 "web_visits_month": 0.20, "age": 0.85, "discount_sensitivity": 0.45, "recency_days": 0.75, "household_size": 0.45},
        signature=("age", "web_visits_month", "recency_days", "spending_score", "total_spend"),
    ),
]

ARCHETYPES_BY_KEY = {a.key: a for a in ARCHETYPES}

# Colour used when a cluster has no confident archetype match.
FALLBACK_COLOR = "#64748B"


# --------------------------------------------------------------------------- #
# External reference context (F04)
# --------------------------------------------------------------------------- #
# Published third-party results on this same dataset. They are shown OUTSIDE the
# ranked leaderboard, with their differing setups stated, because none of them
# share our feature space, cleaning rules or k — so the numbers are context, not
# competitors. Nothing here is computed by this project and nothing here is
# ranked against our models.

EXTERNAL_REFERENCES = [
    {
        "label": "Customer personality analysis and clustering for targeted marketing",
        "silhouette": 0.6186,
        "silhouette_text": "0.6186",
        "k": 2,
        "setup": (
            "K-Means on a reduced feature subset with its own outlier treatment; silhouette "
            "reported at k=2 on that space."
        ),
        "citation": "ResearchGate publication 381943308",
        "url": "https://www.researchgate.net/publication/381943308_Customer_personality_analysis_and_clustering_for_targeted_marketing",
    },
    {
        "label": "Customer Personality Analysis — Kaggle clustering write-up",
        "silhouette": 0.540,
        "silhouette_text": "0.54",
        "k": 4,
        "setup": (
            "K-Means over PCA-reduced components after a different encoding and cleaning "
            "pipeline; silhouette reported at k=4 in the PCA space."
        ),
        "citation": "Prasan N H, Medium",
        "url": "https://medium.com/@prasanNH/customer-personality-analysis-kaggle-clustering-project-1fdad71ea3ec",
    },
]

EXTERNAL_REFERENCE_CAVEAT = (
    "These are third-party figures on the same raw dataset, reproduced here for context only. "
    "They are <b>not comparable</b> to this project's metrics: each uses a different feature subset, "
    "different cleaning and outlier rules, a different number of clusters, and in one case measures "
    "silhouette in a PCA-reduced space rather than the modelling space. Silhouette is not "
    "transferable across feature spaces, so these values are neither ranked nor benchmarked against "
    "the models below. Nothing on this panel is computed by this project."
)


# --------------------------------------------------------------------------- #
# Plain-language glossary
# --------------------------------------------------------------------------- #
# These metrics are the vocabulary of the whole dashboard, so they are explained
# once, here, in words that do not assume a clustering background.

METRIC_GLOSSARY = {
    "silhouette": {
        "name": "Silhouette",
        "short": "How cleanly separated the segments are.",
        "plain": (
            "For each customer, compare how close they sit to their own segment against how close "
            "they sit to the nearest rival segment. Score each customer from -1 to 1, then average "
            "over everyone. 1 means every customer is comfortably inside their own group; 0 means "
            "customers sit right on the boundary between two groups; a negative score means they "
            "are closer to a group they were not assigned to."
        ),
        "reading": (
            "Above 0.5 is strong separation. Around 0.2-0.3 means the groups are real but overlap "
            "heavily at the edges, which is normal for human behavioural data - customers vary "
            "continuously rather than falling into tidy, well-separated clumps."
        ),
        "direction": "Higher is better",
        "range": "-1 to 1",
    },
    "davies_bouldin": {
        "name": "Davies-Bouldin",
        "short": "How much each segment overlaps its closest neighbour.",
        "plain": (
            "For each segment, find its worst-case rival: the one whose members are most similar to "
            "it relative to how spread out the two are. Average that worst case across all segments. "
            "It asks the same question as silhouette from the opposite direction, so the two "
            "usually agree."
        ),
        "reading": "Below 1 is good separation. Around 1.4 means neighbouring segments blur into each other.",
        "direction": "Lower is better",
        "range": "0 upward",
    },
    "calinski_harabasz": {
        "name": "Calinski-Harabasz",
        "short": "Spread between segments versus spread inside them.",
        "plain": (
            "The ratio of how far apart the segment centres are to how spread out the customers are "
            "within each segment. A high score means the centres are far apart and the members are "
            "tightly packed around them."
        ),
        "reading": (
            "It has no fixed scale, so the number is only meaningful when comparing models fitted "
            "on the same data - which is exactly how it is used on the leaderboard."
        ),
        "direction": "Higher is better",
        "range": "0 upward, unbounded",
    },
    "wcss": {
        "name": "Inertia (WCSS)",
        "short": "Total distance from customers to their own segment centre.",
        "plain": (
            "Within-Cluster Sum of Squares: add up the squared distance from every customer to the "
            "centre of the segment they were assigned to. It measures how tightly the model fits, "
            "not how well separated the segments are."
        ),
        "reading": (
            "It always falls as you add more segments - with one segment per customer it reaches "
            "zero - so it can never pick the number of segments on its own. What is useful is the "
            "point where it stops falling steeply: the elbow."
        ),
        "direction": "Lower is better, but only at a fixed number of segments",
        "range": "0 upward",
    },
    "noise_ratio": {
        "name": "Noise",
        "short": "Share of customers the algorithm refused to place in any segment.",
        "plain": (
            "Density-based methods such as DBSCAN are allowed to label a customer as noise rather "
            "than forcing them into a segment. Those customers are excluded when the other metrics "
            "are computed, so a high noise share means the score describes only part of the base."
        ),
        "reading": "Only DBSCAN can produce this; every other method here places all customers.",
        "direction": "Context, not a score",
        "range": "0% to 100%",
    },
}
