"""F10 - the two-proportion test, checked against an independent implementation."""

from __future__ import annotations

import pytest
from statsmodels.stats.proportion import proportions_ztest

from skills_lab.stats import ABInputError, two_proportion_test


def test_matches_statsmodels_pooled_z_test():
    result = two_proportion_test(20_000, 1_000, 20_000, 1_150)
    z, p = proportions_ztest([1_150, 1_000], [20_000, 20_000], alternative="two-sided")
    assert result.z_score == pytest.approx(z, rel=1e-9)
    assert result.p_value == pytest.approx(p, rel=1e-6)


def test_confidence_interval_brackets_the_point_estimate():
    result = two_proportion_test(15_000, 750, 15_000, 900)
    assert result.ci_low_pp < result.absolute_lift_pp < result.ci_high_pp
    assert result.significant is True
    assert result.ci_low_pp > 0


def test_recommendation_uses_direction_magnitude_and_interval_not_p_alone():
    # Significant but tiny: interval reaches below the practical threshold -> hold.
    small = two_proportion_test(
        400_000, 20_000, 400_000, 20_600, minimum_practical_effect_pp=1.0
    )
    assert small.significant is True
    assert "hold" in small.recommendation.lower()

    # Significant, large, whole interval above the threshold -> ship.
    big = two_proportion_test(
        50_000, 2_500, 50_000, 3_500, minimum_practical_effect_pp=0.5
    )
    assert "ship" in big.recommendation.lower()

    # Significant but negative -> keep control.
    worse = two_proportion_test(50_000, 3_500, 50_000, 2_500)
    assert "keep control" in worse.recommendation.lower()

    # Wide, inconclusive interval -> explicitly inconclusive, not "no difference".
    thin = two_proportion_test(300, 30, 300, 36, minimum_practical_effect_pp=0.5)
    assert thin.significant is False
    assert "inconclusive" in thin.recommendation.lower()


def test_post_hoc_power_is_descriptive_only():
    result = two_proportion_test(20_000, 1_000, 20_000, 1_150)
    assert 0.0 <= result.observed_power <= 1.0
    assert "not independent evidence" in result.power_note
    # The decision is driven by significance/interval, not by the power figure.
    assert result.significant == (result.p_value < result.alpha)


def test_degenerate_inputs():
    with pytest.raises(ABInputError):
        two_proportion_test(0, 0, 100, 5)
    with pytest.raises(ABInputError):
        two_proportion_test(100, 200, 100, 5)
    zero = two_proportion_test(500, 0, 500, 0)
    assert zero.z_score == 0.0 and zero.p_value == 1.0
    assert zero.relative_lift_pct == pytest.approx(0.0)


def test_small_count_warning_is_raised():
    result = two_proportion_test(100, 2, 100, 9)
    assert any("normal approximation" in w for w in result.warnings)
