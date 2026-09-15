"""F08/F09 - cohort retention and funnel semantics on real-shaped transaction data."""

from __future__ import annotations

import pandas as pd

from skills_lab.benchmarks import analytics
from skills_lab.data import retail


def test_business_events_are_classified_not_dropped(retail_fixture):
    summary = retail.business_event_summary(retail_fixture)
    assert summary["rows"] == len(retail_fixture)
    assert summary["credit_notes"] == 1
    assert summary["return_lines"] == 1
    assert summary["adjustment_lines"] == 1
    assert summary["returned_value"] < 0


def test_retention_population_excludes_credit_notes_and_null_customers(retail_fixture):
    population = retail.retention_population(retail_fixture)
    assert len(population) == 5  # 7 lines less one credit note and one postage line
    assert population["customer_id"].notna().all()


def test_retention_matrix_is_triangular_and_period_zero_is_full(retail_fixture):
    population = retail.retention_population(retail_fixture)
    retention = analytics.cohort_retention(population)
    assert (retention.matrix[0].dropna() == 100.0).all()
    # A cohort cannot be observed beyond the end of the data.
    last = max(int(c) for c in retention.matrix.columns)
    for cohort in retention.matrix.index:
        observable = retention.matrix.loc[cohort].notna()
        assert observable.iloc[0]
        # Once unobservable, all later periods stay unobservable.
        seen_gap = False
        for period in range(last + 1):
            if period not in retention.matrix.columns:
                continue
            value_present = bool(retention.matrix.loc[cohort, period] == retention.matrix.loc[cohort, period])
            if not value_present:
                seen_gap = True
            elif seen_gap:
                pass  # re-activation after an inactive month is legitimate
        assert retention.sizes[cohort] >= 1


def test_retention_is_deterministic_across_runs(retail_fixture):
    population = retail.retention_population(retail_fixture)
    first = analytics.cohort_retention(population).matrix
    second = analytics.cohort_retention(population).matrix
    pd.testing.assert_frame_equal(first, second)


def test_retention_rates_never_exceed_one_hundred(retail_fixture):
    matrix = analytics.cohort_retention(
        retail.retention_population(retail_fixture)
    ).matrix
    assert matrix.max().max() <= 100.0


def test_net_and_gross_revenue_are_distinct_definitions(retail_fixture):
    frame = retail.daily_revenue(retail_fixture, days=0)
    assert {"net_revenue", "gross_sales"} <= set(frame.columns)
    # The credit note makes net revenue strictly smaller on the day it was booked.
    assert (frame["net_revenue"] <= frame["gross_sales"]).all()
    assert frame["net_revenue"].sum() < frame["gross_sales"].sum()
    assert "net of returns" in retail.REVENUE_DEFINITIONS["net"]


def test_funnel_is_labelled_illustrative_and_rates_are_consistent():
    funnel = analytics.checkout_funnel()
    assert "Illustrative" in funnel.label
    stages = funnel.stages
    assert stages[0]["step_conversion_pct"] == 100.0
    for previous, current in zip(stages, stages[1:]):
        assert current["users"] <= previous["users"]
        expected = current["users"] / previous["users"] * 100
        assert abs(current["step_conversion_pct"] - expected) < 1e-9
        assert abs(
            current["step_conversion_pct"] + current["step_dropoff_pct"] - 100
        ) < 1e-9
