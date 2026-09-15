"""The affinity network may only claim affinities its rules actually assert."""

from __future__ import annotations

from basket import config, graph, mining, rules


def _fixture_rules(corpus, min_support=0.05, max_len=4):
    result = mining.eclat(corpus, min_support, max_len)
    return rules.generate(result, min_confidence=0.0, min_lift=1.0)


def _catalog_map(catalog):
    return {int(r["product_id"]): r for r in catalog.to_dict(orient="records")}


def test_every_edge_comes_from_a_single_to_single_rule(corpus, catalog):
    """The core invariant.

    `{A,B} -> C` says nothing about A and C on their own, so neither a
    first-item-to-first-item edge nor a pairwise decomposition is permitted.
    """
    rule_list = _fixture_rules(corpus)
    assert any(len(r.antecedent) > 1 or len(r.consequent) > 1 for r in rule_list), \
        "fixture must contain multi-item rules for this test to mean anything"

    payload = graph.build(rule_list, _catalog_map(catalog), {}, top_n=50)
    asserted = {
        (r.antecedent[0], r.consequent[0])
        for r in rule_list
        if len(r.antecedent) == 1 and len(r.consequent) == 1
    }
    for edge in payload["edges"]:
        assert (edge["source"], edge["target"]) in asserted


def test_multi_item_rules_are_counted_but_not_drawn(corpus, catalog):
    rule_list = _fixture_rules(corpus)
    payload = graph.build(rule_list, _catalog_map(catalog), {}, top_n=50)
    assert payload["n_rules_total"] == len(rule_list)
    assert payload["n_pairwise_rules_available"] == len(graph.pairwise_rules(rule_list))
    assert payload["n_pairwise_rules_available"] < payload["n_rules_total"]


def test_nodes_carry_display_metadata_but_no_price(corpus, catalog):
    """Prices are illustrative, so they stay out of the analytical artifact."""
    rule_list = _fixture_rules(corpus)
    frequency = {pid: 0.5 for pid in range(1, 11)}
    payload = graph.build(rule_list, _catalog_map(catalog), frequency, top_n=50)
    assert payload["nodes"]
    for node in payload["nodes"]:
        assert "price" not in node and "price_illustrative" not in node
        assert node["department"]
        assert node["color"].startswith("#")
        assert node["marginal_frequency"] == 0.5


def test_layout_is_deterministic_and_on_the_circle(corpus, catalog):
    rule_list = _fixture_rules(corpus)
    catalog_map = _catalog_map(catalog)
    first = graph.build(rule_list, catalog_map, {}, top_n=50)
    second = graph.build(rule_list, catalog_map, {}, top_n=50)
    assert first["nodes"] == second["nodes"]
    for node in first["nodes"]:
        radius = (node["x"] ** 2 + node["y"] ** 2) ** 0.5
        assert abs(radius - config.GRAPH_RADIUS) < 1e-6


def test_parallel_edges_collapse_to_the_strongest(corpus, catalog):
    rule_list = _fixture_rules(corpus)
    payload = graph.build(rule_list, _catalog_map(catalog), {}, top_n=50)
    pairs = [(e["source"], e["target"]) for e in payload["edges"]]
    assert len(pairs) == len(set(pairs))
