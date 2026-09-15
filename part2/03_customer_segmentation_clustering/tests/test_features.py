"""Feature engineering and the persisted-scaler contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from segmentation import config, features


def test_engineered_formulas():
    frame = pd.DataFrame(
        [{"age": 40, "income_k": 60.0, "spending_score": 49.0, "recency_days": 9.0,
          "total_spend": 500.0, "web_visits_month": 8.0, "discount_sensitivity": 0.25,
          "household_size": 3.0}]
    )
    out = features.engineer(frame).iloc[0]
    assert out["discretionary_ratio"] == pytest.approx(60.0 / 50.0)
    assert out["monetary_velocity"] == pytest.approx(500.0 / 10.0)
    assert out["digital_engagement"] == pytest.approx(8.0 * 0.49)
    assert out["deal_affinity"] == pytest.approx(0.25 * 0.51)


def test_divide_by_zero_guards_hold_at_range_floors():
    """Recency 0 and spending score at its floor must not produce inf or NaN."""
    frame = pd.DataFrame(
        [{"age": 18, "income_k": 1.0, "spending_score": 0.0, "recency_days": 0.0,
          "total_spend": 0.0, "web_visits_month": 0.0, "discount_sensitivity": 0.0,
          "household_size": 1.0}]
    )
    out = features.engineer(frame)
    assert np.isfinite(out[config.ENGINEERED_FEATURES].to_numpy()).all()


def test_engineer_rejects_missing_base_attributes():
    with pytest.raises(ValueError, match="missing base attributes"):
        features.engineer(pd.DataFrame({"age": [40]}))


def test_matrix_respects_column_order(clean_sample):
    enriched = features.engineer(clean_sample)
    reversed_columns = list(reversed(config.FEATURE_COLUMNS))
    normal = features.matrix(enriched, config.FEATURE_COLUMNS)
    flipped = features.matrix(enriched, reversed_columns)
    assert np.allclose(normal, flipped[:, ::-1])


def test_scaler_is_applied_never_refitted(clean_sample):
    """A single record scaled by the fitted scaler must match its row in the batch.

    Re-fitting on one record would centre it at zero — the failure mode this
    contract exists to prevent.
    """
    enriched, scaler, scaled = features.build_training_matrix(clean_sample)
    single = enriched.iloc[[0]]
    applied = features.apply_scaler(scaler, single, config.FEATURE_COLUMNS)
    assert np.allclose(applied[0], scaled[0])
    assert not np.allclose(applied[0], 0.0)


def test_training_matrix_is_standardized(clean_sample):
    _, _, scaled = features.build_training_matrix(clean_sample)
    assert scaled.shape[1] == len(config.FEATURE_COLUMNS)
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(scaled.std(axis=0), 1.0, atol=1e-9)
