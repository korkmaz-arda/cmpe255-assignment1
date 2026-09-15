"""Small deterministic fixtures so the suite runs offline, without the parquet cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skills_lab.data import titanic  # noqa: E402


@pytest.fixture
def titanic_fixture() -> pd.DataFrame:
    """A synthetic Titanic-shaped frame: same columns, same dtypes, same missingness."""
    rng = np.random.default_rng(7)
    n = 240
    frame = pd.DataFrame(
        {
            "pclass": rng.choice(["1", "2", "3"], n),
            "sex": rng.choice(["male", "female"], n),
            "age": rng.normal(30, 12, n).clip(1, 80),
            "sibsp": rng.integers(0, 4, n),
            "parch": rng.integers(0, 3, n),
            "fare": rng.exponential(30, n).clip(5, 300),
            "embarked": rng.choice(["S", "C", "Q"], n),
        }
    )
    frame.loc[rng.choice(n, 40, replace=False), "age"] = np.nan
    logit = (
        1.2 * (frame["sex"] == "female")
        - 0.8 * (frame["pclass"] == "3")
        + 0.01 * frame["fare"]
        - 0.02 * frame["age"].fillna(30)
    )
    probability = 1 / (1 + np.exp(-logit))
    frame["survived"] = (rng.random(n) < probability).astype(int)
    frame = titanic.add_derived_features(frame)
    for column in ("pclass", "sex", "embarked"):
        frame[column] = titanic.as_sklearn_object(frame[column].astype("string"))
    return frame[titanic.FEATURES + [titanic.TARGET]]


@pytest.fixture
def ames_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 300
    area = rng.normal(1500, 400, n).clip(500, 4000)
    quality = rng.integers(3, 10, n)
    frame = pd.DataFrame(
        {
            "GrLivArea": area,
            "OverallQual": quality,
            "YearBuilt": rng.integers(1920, 2010, n),
            "TotalBsmtSF": rng.normal(900, 300, n).clip(0, 2500),
            "GarageCars": rng.integers(0, 4, n),
            "FullBath": rng.integers(1, 4, n),
            "Neighborhood": rng.choice(["A", "B", "C", "D"], n),
        }
    )
    log_price = 10.5 + 0.0003 * area + 0.12 * quality + rng.normal(0, 0.12, n)
    frame["SalePrice"] = np.expm1(log_price).round(-2)
    frame["Neighborhood"] = frame["Neighborhood"].astype("string")
    return frame


@pytest.fixture
def fraud_fixture() -> pd.DataFrame:
    """Rare-positive data with the same column names as the real fraud table."""
    rng = np.random.default_rng(3)
    n_neg, n_pos = 4000, 40
    negatives = rng.normal(0, 1, (n_neg, 28))
    positives = rng.normal(2.2, 1.4, (n_pos, 28))
    components = np.vstack([negatives, positives])
    amount = np.concatenate([rng.exponential(40, n_neg), rng.exponential(220, n_pos)])
    labels = np.concatenate([np.zeros(n_neg, int), np.ones(n_pos, int)])
    frame = pd.DataFrame(components, columns=[f"V{i}" for i in range(1, 29)])
    frame["Amount"] = amount
    frame["Class"] = labels
    return frame.sample(frac=1.0, random_state=5).reset_index(drop=True)


@pytest.fixture
def dirty_fixture() -> pd.DataFrame:
    """A small table with real-world-shaped defects and legitimate business events."""
    frame = pd.DataFrame(
        {
            "Invoice": ["A1", "A1", "C2", "A3", "A4", "A4"],
            "StockCode": ["MUG", "mug", "POST", "LAMP", "LAMP", "LAMP"],
            "Description": ["Mug", "Mug", None, "?", "Lamp", "Lamp"],
            "Quantity": [6, 6, -2, 1, 3, 3],
            "Price": [2.5, 2.5, 1.0, 0.0, 4.0, 4.0],
            "Customer ID": [1001.0, 1001.0, None, 1002.0, 1003.0, 1003.0],
            "Country": ["United Kingdom", "United Kingdom", "Unspecified", "France",
                        "France", "France"],
        }
    )
    return frame


@pytest.fixture
def retail_fixture() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2010-01-05", "2010-01-06", "2010-02-10", "2010-02-11", "2010-03-15",
         "2010-03-16", "2010-04-20"]
    )
    return pd.DataFrame(
        {
            "Invoice": ["1", "2", "3", "C4", "5", "6", "7"],
            "StockCode": ["X", "X", "X", "X", "POST", "X", "X"],
            "Description": ["a"] * 7,
            "Quantity": [2, 3, 1, -1, 1, 4, 2],
            "InvoiceDate": dates,
            "Price": [10.0, 10.0, 10.0, 10.0, 5.0, 10.0, 10.0],
            "Customer ID": [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 3.0],
            "Country": ["UK"] * 7,
        }
    )
