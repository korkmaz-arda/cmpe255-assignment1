"""Ames, Iowa house sales (OpenML 42165) prepared for the regression benchmark.

The feature set mirrors the drivers the project's regression workspace is about: size,
quality, age, basement, garage, bathrooms and location.
"""

from __future__ import annotations

import pandas as pd

from skills_lab.data.acquire import load_cached

TARGET = "SalePrice"
NUMERIC_FEATURES = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF",
    "GarageCars",
    "FullBath",
]
CATEGORICAL_FEATURES = ["Neighborhood"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

DISPLAY_NAMES = {
    "GrLivArea": "Above-grade living area (sq ft)",
    "OverallQual": "Overall quality (1-10)",
    "YearBuilt": "Year built",
    "TotalBsmtSF": "Total basement area (sq ft)",
    "GarageCars": "Garage capacity (cars)",
    "FullBath": "Full bathrooms",
    "Neighborhood": "Neighborhood",
    "SalePrice": "Sale price (USD)",
}


def load() -> pd.DataFrame:
    raw = load_cached("ames")
    frame = raw[FEATURES + [TARGET]].copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Neighborhood"] = frame["Neighborhood"].astype("string")
    return frame


def profile() -> dict[str, object]:
    frame = load()
    return {
        "rows": int(len(frame)),
        "neighborhoods": int(frame["Neighborhood"].nunique()),
        "price_median": float(frame[TARGET].median()),
        "price_min": float(frame[TARGET].min()),
        "price_max": float(frame[TARGET].max()),
        # Reported because the source generated basement area as a fraction of living
        # area; in the real data the two are only moderately correlated.
        "area_basement_correlation": float(
            frame["GrLivArea"].corr(frame["TotalBsmtSF"])
        ),
    }
