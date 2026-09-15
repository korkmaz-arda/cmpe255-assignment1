"""Acquisition and preparation of the customer behavioural dataset.

The source project synthesised its data. This reimplementation uses the real
Kaggle *Customer Personality Analysis* dataset (an explicit, user-approved
adaptation) and derives the eight behavioural attributes the rest of the
pipeline expects. Every derivation is documented below and covered by tests.
"""

from __future__ import annotations

import argparse
import io
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from . import config

RAW_REQUIRED_COLUMNS = [
    "ID", "Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome",
    "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases",
    "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth",
    "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response",
]

SPEND_COLUMNS = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
                 "MntSweetProducts", "MntGoldProds"]
CHANNEL_COLUMNS = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
CAMPAIGN_COLUMNS = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4",
                    "AcceptedCmp5", "Response"]
PARTNERED_STATUSES = {"Married", "Together"}

MANUAL_DOWNLOAD_HINT = (
    f"Could not download the dataset automatically.\n"
    f"Download it manually from {config.DATASET_PAGE} and place "
    f"'{config.RAW_CSV_MEMBER}' at {config.RAW_CSV}."
)


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #

def download_raw(force: bool = False) -> "config.Path":
    """Fetch the dataset zip from Kaggle's public endpoint and extract the CSV."""
    if config.RAW_CSV.exists() and not force:
        return config.RAW_CSV

    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    try:
        request = urllib.request.Request(
            config.DATASET_URL, headers={"User-Agent": "cmpe255-segmentation/1.0"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - network
        raise RuntimeError(f"{MANUAL_DOWNLOAD_HINT}\nUnderlying error: {exc}") from exc

    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
        member = next(n for n in archive.namelist() if n.endswith(".csv"))
        config.RAW_CSV.write_bytes(archive.read(member))
    except (zipfile.BadZipFile, StopIteration) as exc:  # pragma: no cover - network
        raise RuntimeError(MANUAL_DOWNLOAD_HINT) from exc

    return config.RAW_CSV


def load_raw(path: "config.Path | None" = None) -> pd.DataFrame:
    """Read the raw tab-separated Kaggle CSV."""
    path = path or config.RAW_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python -m segmentation.data --download`."
        )
    frame = pd.read_csv(path, sep="\t")
    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing expected columns: {missing}")
    return frame


# --------------------------------------------------------------------------- #
# Derivation of the eight behavioural attributes
# --------------------------------------------------------------------------- #

def _percentile_rank(series: pd.Series) -> pd.Series:
    """Rank a series into (0, 1], averaging ties."""
    return series.rank(pct=True, method="average")


def derive_spending_score(frame: pd.DataFrame) -> pd.Series:
    """Derive the 1–100 behavioural spending score.

    The raw dataset has no spending score, so one is constructed. A percentile
    rank of ``total_spend`` would be the obvious choice but measures rho ~= 0.91
    (Spearman) against ``total_spend`` itself, which would hand the clustering
    two near-duplicate monetary dimensions. Purchase-frequency variants are no
    better (rho ~= 0.89-0.93): frequency and spend are intrinsically coupled here.

    The score is therefore built from two *non-monetary* behavioural signals,
    equally weighted:

    * conversion efficiency -- purchases per web visit, i.e. how readily browsing
      turns into buying;
    * campaign responsiveness -- how many of the six marketing campaigns the
      customer accepted.

    This is the semantic analogue of a retailer-assigned engagement/loyalty
    score, and measures rho ~= 0.85 against ``total_spend`` -- correlated, as any
    genuine purchase-behaviour signal must be on this dataset, but no longer a
    near-duplicate of the monetary axis. The realised correlation is recomputed
    and reported by the pipeline rather than assumed.
    """
    channel_purchases = frame[CHANNEL_COLUMNS].sum(axis=1)
    conversion = channel_purchases / (frame["NumWebVisitsMonth"] + 1.0)
    responsiveness = frame[CAMPAIGN_COLUMNS].sum(axis=1)

    blend = 0.5 * _percentile_rank(conversion) + 0.5 * _percentile_rank(responsiveness)
    scaled = _percentile_rank(blend) * 100.0
    return scaled.clip(lower=1.0, upper=100.0)


def derive_discount_sensitivity(frame: pd.DataFrame) -> pd.Series:
    """Share of a customer's purchases that were made with a discount.

    ``NumDealsPurchases`` counts purchases made *with a discount*; the three
    ``Num*Purchases`` columns partition purchases *by channel*. A discounted
    purchase is still made through some channel, so deals are an attribute of the
    channel purchases rather than a fourth channel -- confirmed empirically:
    ``NumDealsPurchases <= Web + Catalog + Store`` holds for 2,237 of the 2,240
    raw rows. Adding deals into the denominator would double-count them and
    systematically understate the ratio, so the denominator is the three channel
    counts only.

    The handful of rows that violate the containment are clipped to 1.0; rows
    with no purchase history at all are dropped upstream in :func:`prepare`,
    because a discount share is undefined for a customer who never purchased.
    """
    channel_purchases = frame[CHANNEL_COLUMNS].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = frame["NumDealsPurchases"] / channel_purchases.replace(0, np.nan)
    return ratio.clip(lower=0.0, upper=1.0)


def derive_household_size(frame: pd.DataFrame) -> pd.Series:
    """Adults in the home plus children and teenagers.

    ``Marital_Status`` carries a few junk levels ("Absurd", "YOLO", "Alone")
    which are treated as not-partnered.
    """
    partnered = frame["Marital_Status"].isin(PARTNERED_STATUSES).astype(int)
    return 1 + partnered + frame["Kidhome"] + frame["Teenhome"]


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw frame and derive the eight behavioural attributes.

    Cleaning rules, all deliberate and all reported by the pipeline:

    * drop rows with a missing ``Income`` (24 rows -- roughly 1% -- imputing a
      household income from the remaining columns would invent the single
      strongest clustering signal);
    * drop income outliers above the 99.5th percentile (removes the 666,666
      data-entry artefact);
    * drop implausible ages, keeping 18-90 (removes the 1893/1899 birth years);
    * drop customers with no purchase history, for whom a discount share and a
      conversion rate are both undefined.
    """
    frame = raw.copy()

    frame = frame.dropna(subset=["Income"])

    income_cap = frame["Income"].quantile(0.995)
    frame = frame[frame["Income"] <= income_cap]

    age = config.REFERENCE_YEAR - frame["Year_Birth"]
    frame = frame[(age >= config.INPUT_RANGES["age"][0]) & (age <= config.INPUT_RANGES["age"][1])]

    channel_purchases = frame[CHANNEL_COLUMNS].sum(axis=1)
    frame = frame[channel_purchases > 0]

    prepared = pd.DataFrame(
        {
            "customer_id": frame["ID"].astype(int),
            "age": (config.REFERENCE_YEAR - frame["Year_Birth"]).astype(float),
            "income_k": (frame["Income"] / 1000.0).astype(float),
            "spending_score": derive_spending_score(frame).astype(float),
            "recency_days": frame["Recency"].astype(float),
            "total_spend": frame[SPEND_COLUMNS].sum(axis=1).astype(float),
            "web_visits_month": frame["NumWebVisitsMonth"].astype(float),
            "discount_sensitivity": derive_discount_sensitivity(frame).astype(float),
            "household_size": derive_household_size(frame).astype(float),
            "education": frame["Education"].astype(str),
        }
    )

    return prepared.dropna().reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def build(force_download: bool = False) -> pd.DataFrame:
    """Download (if needed), clean and cache the analysis-ready dataset."""
    download_raw(force=force_download)
    prepared = prepare(load_raw())
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(config.CLEAN_PARQUET, index=False)
    return prepared


def load_clean(rebuild: bool = False) -> pd.DataFrame:
    """Return the cleaned dataset, building it on first use."""
    if config.CLEAN_PARQUET.exists() and not rebuild:
        return pd.read_parquet(config.CLEAN_PARQUET)
    return build()


def sample(frame: pd.DataFrame, n: int | None, seed: int = config.DEFAULT_SEED) -> pd.DataFrame:
    """Deterministically subsample rows; ``None`` or an oversized n returns all."""
    if n is None or n >= len(frame):
        return frame.reset_index(drop=True)
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def quality_report(clean: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    """Summarise the cleaned dataset for the UI's data-understanding panels."""
    report = {
        "n_customers": int(len(clean)),
        "n_attributes": len(config.BASE_FEATURES),
        "spending_score_vs_total_spend_spearman": float(
            clean["spending_score"].corr(clean["total_spend"], method="spearman")
        ),
        "attributes": {
            col: {
                "min": float(clean[col].min()),
                "max": float(clean[col].max()),
                "mean": float(clean[col].mean()),
                "label": config.ATTRIBUTE_LABELS[col],
                "meaning": config.ATTRIBUTE_MEANINGS[col],
            }
            for col in config.BASE_FEATURES
        },
    }
    if raw is not None:
        report["n_raw_rows"] = int(len(raw))
        report["n_dropped"] = int(len(raw) - len(clean))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the customer dataset.")
    parser.add_argument("--download", action="store_true", help="fetch the raw dataset if absent")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args()

    if args.download or args.force:
        download_raw(force=args.force)

    raw = load_raw()
    clean = build()
    report = quality_report(clean, raw)
    print(f"Raw rows:     {report['n_raw_rows']}")
    print(f"Clean rows:   {report['n_customers']} ({report['n_dropped']} dropped)")
    print(f"spending_score vs total_spend (Spearman): "
          f"{report['spending_score_vs_total_spend_spearman']:.3f}")
    print(f"Written to:   {config.CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
