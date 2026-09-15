"""Titanic passenger data (OpenML 40945) prepared for the classification benchmark.

Feature engineering here is deliberately minimal and, importantly, outcome-independent:
``boat`` and ``body`` are excluded because they record what happened *after* the sinking
and would leak the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from skills_lab.data.acquire import load_cached

TARGET = "survived"
NUMERIC_FEATURES = ["age", "fare", "sibsp", "parch", "family_size"]
CATEGORICAL_FEATURES = ["pclass", "sex", "embarked", "is_alone"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns that encode the outcome itself and must never become features.
LEAKING_COLUMNS = ["boat", "body", "name", "ticket", "cabin", "home.dest"]


def as_sklearn_object(series: pd.Series) -> pd.Series:
    """Cast a pandas string column to object dtype with np.nan for missing values.

    scikit-learn's imputers cannot evaluate pandas' pd.NA in a boolean context, so the
    nullable string dtype has to be flattened before it reaches a ColumnTransformer.
    """
    return series.astype(object).where(series.notna(), np.nan)


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the two engineered household features used by the model and the predictor UI."""
    out = frame.copy()
    out["family_size"] = out["sibsp"].astype("int64") + out["parch"].astype("int64") + 1
    out["is_alone"] = (out["family_size"] == 1).map({True: "alone", False: "with family"})
    return out


def load() -> pd.DataFrame:
    raw = load_cached("titanic")
    frame = raw.drop(columns=[c for c in LEAKING_COLUMNS if c in raw.columns])
    frame[TARGET] = frame[TARGET].astype("int64")
    frame["pclass"] = as_sklearn_object(frame["pclass"].astype("int64").astype("string"))
    frame["sex"] = as_sklearn_object(frame["sex"].astype("string"))
    frame["embarked"] = as_sklearn_object(frame["embarked"].astype("string"))
    frame = add_derived_features(frame)
    return frame[FEATURES + [TARGET]]


def profile() -> dict[str, object]:
    """Small factual description used by the UI banner (no invented numbers)."""
    frame = load()
    return {
        "rows": int(len(frame)),
        "positive_rate": float(frame[TARGET].mean()),
        "age_missing_rate": float(frame["age"].isna().mean()),
        "fare_missing_rate": float(frame["fare"].isna().mean()),
    }
