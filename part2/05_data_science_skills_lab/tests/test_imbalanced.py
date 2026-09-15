"""F06/F07 - threshold discipline on rare positives."""

from __future__ import annotations

import numpy as np
import pytest

from skills_lab.benchmarks import imbalanced


def test_split_is_three_way_and_stratified(fraud_fixture):
    result = imbalanced.run(fraud_fixture)
    assert result.train_rows + result.validation_rows + result.test_rows == result.rows
    assert result.validation_rows > 0
    assert 0 < result.positive_rate < 0.05


def test_threshold_is_selected_on_validation_not_test(fraud_fixture):
    result = imbalanced.run(fraud_fixture)
    # The selected threshold maximises F1 over the validation sweep, by construction.
    best = max(result.threshold_sweep, key=lambda row: row["f1"])
    assert result.selected_threshold == pytest.approx(best["threshold"])
    assert result.selected_validation_f1 == pytest.approx(best["f1"])

    # Recomputing on validation reproduces the reported validation F1 ...
    measured = imbalanced.threshold_metrics(
        result.validation_labels, result.validation_scores, result.selected_threshold
    )
    assert measured["f1"] == pytest.approx(result.selected_validation_f1, abs=1e-9)
    # ... and the reported final F1 is a *different*, test-partition measurement.
    assert result.weighted_selected_test.threshold == pytest.approx(
        result.selected_threshold
    )


def test_candidate_thresholds_are_not_restricted_to_a_fixed_grid(fraud_fixture):
    result = imbalanced.run(fraud_fixture)
    thresholds = [row["threshold"] for row in result.threshold_sweep]
    assert len(thresholds) > 100  # every PR-curve cutoff, not 30 grid points
    assert min(thresholds) < 0  # log-odds below the default cutoff are reachable
    assert thresholds == sorted(thresholds)


def test_class_weighting_raises_recall_and_average_precision_is_primary(fraud_fixture):
    result = imbalanced.run(fraud_fixture)
    assert result.weighted_test.recall >= result.baseline_test.recall
    assert result.payload()["primary_metric"] == "average_precision (PR-AUC)"
    # Both models are treated identically apart from class_weight.
    assert result.baseline_test.class_weight == "none"
    assert result.weighted_test.class_weight == "balanced"


def test_threshold_metrics_trade_recall_against_precision(fraud_fixture):
    result = imbalanced.run(fraud_fixture)
    low, high = np.quantile(result.validation_scores, [0.90, 0.999])
    loose = imbalanced.threshold_metrics(
        result.validation_labels, result.validation_scores, low
    )
    strict = imbalanced.threshold_metrics(
        result.validation_labels, result.validation_scores, high
    )
    assert loose["recall"] >= strict["recall"]
    assert loose["flagged"] >= strict["flagged"]


def test_review_budget_thresholds_are_monotone(fraud_fixture):
    result = imbalanced.run(fraud_fixture)
    budgets = imbalanced.review_budget_thresholds(result.validation_scores, points=20)
    rates = [b["review_rate"] for b in budgets]
    thresholds = [b["threshold"] for b in budgets]
    assert rates == sorted(rates)
    assert thresholds == sorted(thresholds, reverse=True)
    assert imbalanced.as_probability(0.0) == pytest.approx(0.5)


def test_selected_threshold_is_on_the_slider_so_the_default_reports_it(fraud_fixture):
    """The workbench must open on the selected cutoff, not the nearest grid point."""
    result = imbalanced.run(fraud_fixture)
    budgets = imbalanced.review_budget_thresholds(
        result.validation_scores, points=20, include=result.selected_threshold
    )
    thresholds = [b["threshold"] for b in budgets]
    assert any(t == pytest.approx(result.selected_threshold) for t in thresholds)
    rates = [b["review_rate"] for b in budgets]
    assert rates == sorted(rates)
    match = next(
        b for b in budgets if b["threshold"] == pytest.approx(result.selected_threshold)
    )
    measured = imbalanced.threshold_metrics(
        result.validation_labels, result.validation_scores, match["threshold"]
    )
    assert measured["f1"] == pytest.approx(result.selected_validation_f1, abs=1e-9)
