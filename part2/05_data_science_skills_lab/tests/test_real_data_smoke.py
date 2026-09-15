"""Smoke tests against the real cached datasets. Skipped when the cache is absent."""

from __future__ import annotations

import pytest

from skills_lab.data import acquire, ames, fraud, titanic

pytestmark = pytest.mark.skipif(
    bool(acquire.missing_datasets()),
    reason="run python scripts/prepare_data.py to enable real-data smoke tests",
)


def test_titanic_shape_and_no_outcome_leaking_columns():
    frame = titanic.load()
    assert len(frame) > 1_000
    assert "boat" not in frame.columns and "body" not in frame.columns
    assert frame["age"].isna().mean() > 0.1  # real missingness is preserved, not filled


def test_ames_features_present():
    frame = ames.load()
    assert len(frame) == 1_460
    assert frame["SalePrice"].min() > 0
    assert frame["Neighborhood"].nunique() > 5


def test_fraud_base_rate_is_the_real_one():
    profile = fraud.profile()
    assert profile["rows"] > 280_000
    assert 0.001 < profile["positive_rate"] < 0.003  # ~0.17%
    assert profile["positives"] == 492
