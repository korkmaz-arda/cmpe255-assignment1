"""Feature engineering and standardization (F02).

Four ratio/interaction features are derived from the eight base attributes,
giving the 12-column matrix used for all clustering and inference. The scaler is
fitted exactly once during training, persisted alongside the model and the
ordered column list, and thereafter only ever *applied* — re-fitting a scaler on
a single inference record would silently destroy every prediction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import config

# Formulas surfaced in the UI so the engineered signals are self-describing.
FEATURE_FORMULAS = {
    "discretionary_ratio": "income_k / (spending_score + 1)",
    "monetary_velocity": "total_spend / (recency_days + 1)",
    "digital_engagement": "web_visits_month * (spending_score / 100)",
    "deal_affinity": "discount_sensitivity * (1 - spending_score / 100)",
}

FEATURE_MEANINGS = {
    "discretionary_ratio": "Earning power relative to engagement — high means untapped capacity.",
    "monetary_velocity": "How fast money is flowing in right now: spend per day since the last order.",
    "digital_engagement": "Browsing weighted by how readily that browsing converts.",
    "deal_affinity": "Discount reliance concentrated among the least engaged customers.",
}


def engineer(frame: pd.DataFrame) -> pd.DataFrame:
    """Append the four interaction features to a frame of base attributes.

    The ``+ 1`` denominators are deliberate divide-by-zero guards: recency can be
    0 for a customer who ordered today, and the spending score floor is 1 but the
    guard keeps the transform total for any input inside the validated range.
    """
    missing = [c for c in config.BASE_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Cannot engineer features, missing base attributes: {missing}")

    out = frame.copy()
    out["discretionary_ratio"] = out["income_k"] / (out["spending_score"] + 1.0)
    out["monetary_velocity"] = out["total_spend"] / (out["recency_days"] + 1.0)
    out["digital_engagement"] = out["web_visits_month"] * (out["spending_score"] / 100.0)
    out["deal_affinity"] = out["discount_sensitivity"] * (1.0 - out["spending_score"] / 100.0)
    return out


def matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> np.ndarray:
    """Extract the ordered feature matrix. Column order is part of the contract."""
    columns = columns or config.FEATURE_COLUMNS
    return frame[columns].to_numpy(dtype=float)


def fit_scaler(frame: pd.DataFrame, columns: list[str] | None = None) -> tuple[StandardScaler, np.ndarray]:
    """Fit the z-score scaler on the training matrix and return it with the result."""
    columns = columns or config.FEATURE_COLUMNS
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix(frame, columns))
    return scaler, scaled


def apply_scaler(scaler: StandardScaler, frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Apply a *previously fitted* scaler. Never re-fits — that is the point."""
    return scaler.transform(matrix(frame, columns))


def build_training_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler, np.ndarray]:
    """Engineer features and standardize, returning the enriched frame too."""
    enriched = engineer(frame)
    scaler, scaled = fit_scaler(enriched)
    return enriched, scaler, scaled
