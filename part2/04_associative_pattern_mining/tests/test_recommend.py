"""Cross-sell recommendation: ranking, attribution and the empty states."""

from __future__ import annotations

import pytest

from basket import mining, recommend, rules


@pytest.fixture
def served(corpus, catalog):
    result = mining.eclat(corpus, 0.05, 4)
    rule_list = rules.generate(result, min_confidence=0.3, min_lift=1.0)
    catalog_map = {int(r["product_id"]): r for r in catalog.to_dict(orient="records")}
    return rule_list, catalog_map


def test_rules_fire_only_on_a_full_antecedent_match(served):
    rule_list, catalog_map = served
    result = recommend.recommend([1], rule_list, catalog_map)
    suggested = {s.product_id for s in result.suggestions}
    assert 1 not in suggested                      # never suggest what is already there
    assert result.n_rules_fired > 0


def test_ranking_uses_lift_times_confidence(served):
    rule_list, catalog_map = served
    result = recommend.recommend([1, 2], rule_list, catalog_map)
    scores = [s.score for s in result.suggestions]
    assert scores == sorted(scores, reverse=True)
    for suggestion in result.suggestions:
        assert suggestion.score == pytest.approx(suggestion.lift * suggestion.confidence)


def test_headline_rule_is_the_rule_that_produced_the_score(served):
    """Guard against the source's defect, where the badge could come from another rule."""
    rule_list, catalog_map = served
    result = recommend.recommend([6, 7], rule_list, catalog_map)
    assert result.suggestions
    for suggestion in result.suggestions:
        matching = [
            r for r in rule_list
            if set(r.antecedent) <= {6, 7}
            and suggestion.product_id in r.consequent
            and r.lift * r.confidence == pytest.approx(suggestion.score)
        ]
        assert matching, "score must trace to a real firing rule"
        rule = matching[0]
        assert suggestion.lift == pytest.approx(rule.lift)
        assert suggestion.confidence == pytest.approx(rule.confidence)


def test_reason_states_confidence_as_a_conditional_rate(served):
    rule_list, catalog_map = served
    result = recommend.recommend([1], rule_list, catalog_map)
    reason = result.suggestions[0].reason
    assert reason.startswith("Of the baskets that already contain")
    assert "also contain" in reason


def test_empty_basket_and_no_match_are_honest_states(served):
    rule_list, catalog_map = served
    empty = recommend.recommend([], rule_list, catalog_map)
    assert empty.is_empty
    assert empty.n_rules_fired == 0
    assert empty.basket_value_illustrative == 0.0

    # Product 5 appears only alongside product 4, and 4 -> 5 rules need 4 present.
    lonely = recommend.recommend([9999], rule_list, catalog_map)
    assert lonely.is_empty
    assert lonely.addon_value_illustrative == 0.0


def test_limit_and_supporting_rules(served):
    rule_list, catalog_map = served
    result = recommend.recommend([6, 7, 8], rule_list, catalog_map, limit=2)
    assert len(result.suggestions) <= 2
    for suggestion in result.suggestions:
        assert 1 <= len(suggestion.supporting_rules) <= 3


def test_economics_sum_illustrative_prices(served):
    rule_list, catalog_map = served
    result = recommend.recommend([1, 2], rule_list, catalog_map)
    expected = catalog_map[1]["price_illustrative"] + catalog_map[2]["price_illustrative"]
    assert result.basket_value_illustrative == pytest.approx(expected, abs=0.01)
    top3 = sum(s.price_illustrative for s in result.suggestions[:3])
    assert result.addon_value_illustrative == pytest.approx(top3, abs=0.01)
