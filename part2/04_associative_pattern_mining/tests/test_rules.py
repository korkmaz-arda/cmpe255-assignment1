"""Rule generation and the five interestingness metrics."""

from __future__ import annotations

import pytest

from basket import config, mining, rules


def test_metrics_match_hand_computation(corpus):
    """1 -> 2, worked out by hand from the fixture.

    support(1) = 65/105, support(2) = 55/105, support({1,2}) = 45/105.
    """
    result = mining.apriori(corpus, min_support=0.05, max_len=4)
    generated = rules.generate(result, min_confidence=0.0, min_lift=0.0)
    rule = next(r for r in generated if r.antecedent == (1,) and r.consequent == (2,))

    s_a, s_c, s_u = 65 / 105, 55 / 105, 45 / 105
    assert rule.support == pytest.approx(s_u)
    assert rule.confidence == pytest.approx(s_u / s_a)
    assert rule.lift == pytest.approx(s_u / (s_a * s_c))
    assert rule.leverage == pytest.approx(s_u - s_a * s_c)
    assert rule.conviction == pytest.approx((1 - s_c) / (1 - s_u / s_a))


def test_confidence_is_a_conditional_rate(corpus):
    """Confidence divides by the antecedent's support, not by everything."""
    result = mining.apriori(corpus, min_support=0.05, max_len=4)
    generated = rules.generate(result, 0.0, 0.0)
    forward = next(r for r in generated if r.antecedent == (1,) and r.consequent == (2,))
    backward = next(r for r in generated if r.antecedent == (2,) and r.consequent == (1,))
    # Same support both ways, different confidence: that asymmetry is the point.
    assert forward.support == pytest.approx(backward.support)
    assert forward.confidence != pytest.approx(backward.confidence)
    assert forward.confidence == pytest.approx(45 / 65)
    assert backward.confidence == pytest.approx(45 / 55)


def test_conviction_is_finite_when_confidence_is_one(corpus):
    """The 5-itemset gives a rule that always holds; conviction must not divide by zero."""
    result = mining.eclat(corpus, min_support=0.10, max_len=5)
    generated = rules.generate(result, 0.0, 0.0)
    certain = [r for r in generated if r.confidence == pytest.approx(1.0)]
    assert certain
    for rule in certain:
        assert rule.conviction == pytest.approx(
            (1 - rule.consequent_support) / config.EPSILON
        )
        assert rule.conviction < float("inf")


def test_four_product_antecedents_appear_at_length_five(corpus):
    result = mining.eclat(corpus, min_support=0.10, max_len=5)
    generated = rules.generate(result, 0.5, 1.0)
    assert max(len(r.antecedent) for r in generated) == 4


def test_filters_and_ranking(corpus):
    result = mining.apriori(corpus, min_support=0.05, max_len=4)
    generated = rules.generate(result, min_confidence=0.6, min_lift=1.2)
    assert generated
    for rule in generated:
        assert rule.confidence >= 0.6
        assert rule.lift >= 1.2
    lifts = [r.lift for r in generated]
    assert lifts == sorted(lifts, reverse=True)
    # Ties on lift are broken by confidence descending.
    for earlier, later in zip(generated, generated[1:]):
        if earlier.lift == pytest.approx(later.lift):
            assert earlier.confidence >= later.confidence


def test_partitions_with_an_infrequent_side_are_skipped(corpus):
    """A marginal that was never counted cannot be divided by."""
    result = mining.eclat(corpus, min_support=0.13, max_len=4)
    generated = rules.generate(result, 0.0, 0.0)
    for rule in generated:
        assert rule.antecedent in result.itemsets
        assert rule.consequent in result.itemsets


def test_multi_item_consequents_are_produced(corpus):
    result = mining.eclat(corpus, min_support=0.10, max_len=5)
    generated = rules.generate(result, 0.0, 1.0)
    assert any(len(r.consequent) > 1 for r in generated)


def test_render_joins_names_without_making_them_keys(corpus):
    result = mining.apriori(corpus, min_support=0.05, max_len=4)
    generated = rules.generate(result, 0.6, 1.2)
    names = {1: "Alpha Apples", 2: "Beta Bananas", 3: "Gamma Grapes"}
    records = rules.render_all(generated, names)
    assert all("➔" in record["text"] for record in records)
    # Ids survive the round trip; the text is decoration.
    restored = [rules.from_record(record) for record in records]
    assert restored == generated
