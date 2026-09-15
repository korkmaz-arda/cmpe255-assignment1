"""Attribute derivation and cleaning rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from segmentation import config, data


def test_prepare_emits_all_base_attributes(clean_sample):
    for column in config.BASE_FEATURES + ["customer_id"]:
        assert column in clean_sample.columns
    assert len(clean_sample) > 100


def test_rows_with_missing_income_are_dropped(raw_sample, clean_sample):
    dropped_id = int(raw_sample.loc[0, "ID"])
    assert raw_sample.loc[0, "Income"] != raw_sample.loc[0, "Income"]  # is NaN
    assert dropped_id not in set(clean_sample["customer_id"])


def test_income_outlier_is_dropped(raw_sample, clean_sample):
    assert raw_sample.loc[1, "Income"] == 666666
    assert int(raw_sample.loc[1, "ID"]) not in set(clean_sample["customer_id"])
    assert clean_sample["income_k"].max() < 200


def test_implausible_age_is_dropped(raw_sample, clean_sample):
    assert raw_sample.loc[2, "Year_Birth"] == 1893
    assert int(raw_sample.loc[2, "ID"]) not in set(clean_sample["customer_id"])
    low, high = config.INPUT_RANGES["age"]
    assert clean_sample["age"].between(low, high).all()


def test_customers_with_no_purchase_history_are_dropped(raw_sample, clean_sample):
    assert int(raw_sample.loc[3, "ID"]) not in set(clean_sample["customer_id"])


def test_discount_sensitivity_uses_channel_denominator_only():
    """Deals are a discount attribute of channel purchases, not a fourth channel.

    A customer with 2 deal purchases out of 10 channel purchases has a 20% deal
    share. Adding deals into the denominator would report 2/12 = 16.7%.
    """
    frame = pd.DataFrame(
        {
            "NumDealsPurchases": [2],
            "NumWebPurchases": [4],
            "NumCatalogPurchases": [3],
            "NumStorePurchases": [3],
        }
    )
    assert data.derive_discount_sensitivity(frame).iloc[0] == pytest.approx(0.2)


def test_discount_sensitivity_is_clipped_when_deals_exceed_channels(raw_sample, clean_sample):
    """The three raw rows that violate the containment must not produce a ratio > 1."""
    assert clean_sample["discount_sensitivity"].between(0.0, 1.0).all()
    edge_id = int(raw_sample.loc[4, "ID"])
    if edge_id in set(clean_sample["customer_id"]):
        row = clean_sample.loc[clean_sample["customer_id"] == edge_id].iloc[0]
        assert row["discount_sensitivity"] == pytest.approx(1.0)


def test_household_size_treats_junk_marital_status_as_unpartnered():
    frame = pd.DataFrame(
        {
            "Marital_Status": ["Married", "Together", "Single", "YOLO", "Absurd", "Alone"],
            "Kidhome": [1, 0, 0, 1, 0, 0],
            "Teenhome": [0, 1, 0, 0, 0, 0],
        }
    )
    assert list(data.derive_household_size(frame)) == [3, 3, 1, 2, 1, 1]


def test_spending_score_is_bounded(clean_sample):
    assert clean_sample["spending_score"].between(1.0, 100.0).all()


def test_spending_score_is_not_a_duplicate_of_total_spend(clean_sample):
    """Regression guard on the near-duplicate-signal problem.

    A percentile rank of total spend measures rho ~= 0.91 against total spend.
    The engagement-based derivation must stay meaningfully below that, or the
    clustering is handed two copies of the monetary axis.
    """
    rho = clean_sample["spending_score"].corr(clean_sample["total_spend"], method="spearman")
    assert 0.5 < rho < 0.90, f"spending_score correlation with total_spend drifted to {rho:.3f}"


def test_preparation_is_deterministic(raw_sample):
    first = data.prepare(raw_sample)
    second = data.prepare(raw_sample)
    pd.testing.assert_frame_equal(first, second)


def test_sample_is_deterministic(clean_sample):
    a = data.sample(clean_sample, 50, seed=1)
    b = data.sample(clean_sample, 50, seed=1)
    c = data.sample(clean_sample, 50, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["customer_id"].equals(c["customer_id"])
    assert len(data.sample(clean_sample, 10_000, seed=1)) == len(clean_sample)


def test_no_nulls_survive(clean_sample):
    assert not clean_sample[config.BASE_FEATURES].isna().any().any()
    assert np.isfinite(clean_sample[config.BASE_FEATURES].to_numpy()).all()
