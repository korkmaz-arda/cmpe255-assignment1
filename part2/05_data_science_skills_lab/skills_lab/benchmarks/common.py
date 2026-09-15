"""Shared pieces for the supervised benchmarks.

The single most important rule this project demonstrates lives here: every transformer is
assembled inside a Pipeline so that ``fit`` only ever sees the training partition. No
imputation statistic, scaling statistic or category vocabulary is computed from data the
model is later scored on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(name) for name in preprocessor.get_feature_names_out()]


@dataclass
class Importance:
    feature: str
    importance: float
    share_pct: float


def top_importances(
    names: list[str], values: np.ndarray, limit: int = 8
) -> list[Importance]:
    """Model importances as a share of total importance, largest first."""
    total = float(np.sum(values)) or 1.0
    order = np.argsort(values)[::-1][:limit]
    return [
        Importance(
            feature=names[i],
            importance=float(values[i]),
            share_pct=float(values[i]) / total * 100,
        )
        for i in order
    ]


def downsample_curve(
    x: np.ndarray, y: np.ndarray, points: int = 120
) -> list[dict[str, float]]:
    """Thin a curve for plotting while keeping its first and last points."""
    n = len(x)
    if n <= points:
        idx = np.arange(n)
    else:
        idx = np.unique(np.linspace(0, n - 1, points).astype(int))
    return [{"x": float(x[i]), "y": float(y[i])} for i in idx]


def sample_rows(frame: pd.DataFrame, limit: int = 8) -> list[dict]:
    """JSON-safe preview rows; NaN becomes an explicit 'missing' marker."""
    preview = frame.head(limit).copy()
    records: list[dict] = []
    for record in preview.to_dict(orient="records"):
        clean = {}
        for key, value in record.items():
            if pd.isna(value):
                clean[key] = "missing"
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating, float)):
                clean[key] = round(float(value), 4)
            else:
                clean[key] = str(value)
        records.append(clean)
    return records
