"""F11 - rate-based quality scoring, and defects kept distinct from business events."""

from __future__ import annotations

import pandas as pd
import pytest

from skills_lab.benchmarks import quality
from skills_lab.config import HEALTH_PASS_MIN, HEALTH_WARNING_MIN, QUALITY_WEIGHTS


def test_weights_sum_to_one():
    assert sum(QUALITY_WEIGHTS.values()) == pytest.approx(1.0)


def test_dimensions_are_rates_and_composite_is_their_weighted_sum(dirty_fixture):
    result = quality.run(dirty_fixture)
    for value in (result.completeness, result.validity, result.uniqueness, result.consistency):
        assert 0.0 <= value <= 100.0
    expected = sum(
        getattr(result, name) * weight for name, weight in QUALITY_WEIGHTS.items()
    )
    assert abs(result.score - expected) < 1e-9
    assert 0.0 <= result.score <= 100.0


def test_score_is_scale_invariant(dirty_fixture):
    """The same defect rate must score the same on a big table as on a small one."""
    small = quality.run(dirty_fixture)
    big = quality.run(pd.concat([dirty_fixture] * 50, ignore_index=True))
    # Duplicating the table changes uniqueness (by design) but not the other dimensions.
    assert big.completeness == small.completeness
    assert round(big.validity, 6) == round(small.validity, 6)


def test_health_state_comes_from_scores_not_issue_counts():
    clean = pd.DataFrame({"Price": [1.0, 2.0, 3.0], "Country": ["UK", "UK", "UK"]})
    report = quality.profile_column("Price", clean["Price"])
    assert report.health == "PASS"
    assert report.score >= HEALTH_PASS_MIN

    mostly_null = pd.Series([1.0, None, None, None, None], name="Price")
    bad = quality.profile_column("Price", mostly_null)
    assert bad.health == "CRITICAL"

    # Many benign issues on a large column must not produce CRITICAL.
    values = pd.Series([1.0] * 1000 + [None] * 5, name="Price")
    ok = quality.profile_column("Price", values)
    assert ok.issues  # counts are still reported as diagnostics
    assert ok.health == "PASS"
    assert ok.score > HEALTH_WARNING_MIN


def test_returns_and_outliers_are_review_flags_not_defects(dirty_fixture):
    result = quality.run(dirty_fixture)
    quantity = next(c for c in result.column_reports if c.column == "Quantity")
    assert quantity.issues == []  # a negative quantity is a return, not a defect
    assert any("returns/cancellations" in flag for flag in quantity.review_flags)
    assert result.business_events["credit_notes"] >= 1
    assert result.business_events["return_lines"] >= 1


def test_real_defects_are_detected(dirty_fixture):
    result = quality.run(dirty_fixture)
    by_column = {c.column: c for c in result.column_reports}
    assert by_column["Description"].nulls == 1
    assert any("missing" in issue for issue in by_column["Description"].issues)
    # '?' placeholder text and a zero price are genuine validity breaches.
    assert by_column["Description"].invalid >= 1
    assert by_column["Price"].invalid >= 1
    # 'MUG' vs 'mug' is an inconsistency, and 'Unspecified' is not a real country.
    assert by_column["StockCode"].inconsistent >= 1
    assert by_column["Country"].invalid >= 1
    assert result.duplicate_rows == 1  # the repeated LAMP line


def test_identifier_columns_are_not_scanned_for_numeric_outliers(dirty_fixture):
    result = quality.run(dirty_fixture)
    stock = next(c for c in result.column_reports if c.column == "StockCode")
    assert not any("outlier" in flag for flag in stock.review_flags)
