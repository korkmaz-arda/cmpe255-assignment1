"""Product-affinity network.

An edge between two products claims those two products have an affinity. Only a
rule whose antecedent *and* consequent are each a single product actually asserts
that, so the network is built from those rules alone.

The source project drew one edge between a rule's first antecedent item and its
first consequent item while showing the whole rule in the tooltip, which claims
affinities the rule never states. Decomposing a multi-item rule into pairwise
edges has the same problem: `{A, B} -> C` says nothing about A and C on their
own. Neither is reproduced. Multi-item rules keep their full meaning in the
recommender and the rules explorer.

The layout is a fixed circle grouped by department, computed here so the UI does
no layout work. It is circular, not force-directed.
"""

from __future__ import annotations

import math

from . import config
from .rules import Rule


def pairwise_rules(rules: list[Rule]) -> list[Rule]:
    """Rules that assert an affinity between exactly two products."""
    return [r for r in rules if len(r.antecedent) == 1 and len(r.consequent) == 1]


def build(
    rules: list[Rule],
    catalog: dict[int, dict],
    item_frequency: dict[int, float],
    top_n: int = config.GRAPH_TOP_RULES,
) -> dict:
    """Node-link payload for the affinity network.

    ``catalog`` maps product_id to its display metadata. Illustrative prices are
    deliberately not written into this artifact: graph topology, edges, positions
    and ranking are price-independent, and the UI joins the price from the
    catalog when it renders a tooltip.
    """
    candidates = pairwise_rules(rules)[:top_n]

    # Collapse parallel edges: keep the strongest rule for a product pair, but
    # remember how many rules ran between them.
    strongest: dict[tuple[int, int], Rule] = {}
    multiplicity: dict[tuple[int, int], int] = {}
    for rule in candidates:
        key = (rule.antecedent[0], rule.consequent[0])
        multiplicity[key] = multiplicity.get(key, 0) + 1
        current = strongest.get(key)
        if current is None or rule.lift > current.lift:
            strongest[key] = rule

    node_ids = sorted({item for pair in strongest for item in pair})
    positions = _circular_layout(node_ids, catalog)

    nodes = []
    for product_id in node_ids:
        meta = catalog.get(product_id, {})
        x, y = positions[product_id]
        nodes.append(
            {
                "product_id": product_id,
                "product_name": meta.get("product_name", str(product_id)),
                "department": meta.get("department", "unknown"),
                "department_id": int(meta.get("department_id", -1)),
                "color": meta.get("color", config.FALLBACK_COLOR),
                "marginal_frequency": float(item_frequency.get(product_id, 0.0)),
                "x": x,
                "y": y,
            }
        )

    edges = []
    for (source, target), rule in strongest.items():
        edges.append(
            {
                "source": source,
                "target": target,
                "source_name": catalog.get(source, {}).get("product_name", str(source)),
                "target_name": catalog.get(target, {}).get("product_name", str(target)),
                "lift": rule.lift,
                "confidence": rule.confidence,
                "support": rule.support,
                "text": (
                    f"{catalog.get(source, {}).get('product_name', source)} ➔ "
                    f"{catalog.get(target, {}).get('product_name', target)}"
                ),
                "rule_count": multiplicity[(source, target)],
            }
        )
    edges.sort(key=lambda e: (-e["lift"], e["source"], e["target"]))

    return {
        "layout": "circular, grouped by department (precomputed; not force-directed)",
        "edge_semantics": (
            "Each edge is one single-product ➔ single-product rule. Rules with "
            "multiple items on either side are not drawn, because they assert "
            "nothing about any individual pair."
        ),
        "nodes": nodes,
        "edges": edges,
        "n_pairwise_rules_available": len(pairwise_rules(rules)),
        "n_rules_total": len(rules),
        "visible_edges": config.GRAPH_VISIBLE_EDGES,
    }


def _circular_layout(node_ids: list[int], catalog: dict[int, dict]) -> dict[int, tuple[float, float]]:
    """Fixed circle, products grouped by department so neighbours share a colour."""
    ordered = sorted(
        node_ids,
        key=lambda pid: (
            int(catalog.get(pid, {}).get("department_id", 999)),
            catalog.get(pid, {}).get("product_name", str(pid)),
        ),
    )
    positions: dict[int, tuple[float, float]] = {}
    count = max(len(ordered), 1)
    for index, product_id in enumerate(ordered):
        angle = 2.0 * math.pi * index / count
        positions[product_id] = (
            round(config.GRAPH_RADIUS * math.cos(angle), 6),
            round(config.GRAPH_RADIUS * math.sin(angle), 6),
        )
    return positions
