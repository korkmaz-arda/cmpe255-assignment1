"""Shopper / merchandiser workspace: build a basket, see what to suggest and why."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import glossary, state
from basket import config


def render(engine) -> None:
    basket = state.ensure_basket(engine)

    recommendation = engine.recommend(basket)

    _problem_banner(engine, recommendation)
    left, right = st.columns([5, 6], gap="large")
    with left:
        _basket_panel(engine, basket)
    with right:
        _economics(recommendation)
        _recommendations(engine, recommendation)

    st.divider()
    _metric_strip()
    st.divider()
    _affinity_graph(engine)


# --------------------------------------------------------------------------- #
# Banner and headline figures
# --------------------------------------------------------------------------- #

def _problem_banner(engine, recommendation) -> None:
    headline = engine.headline
    st.markdown(
        "#### Which product should we suggest next, and what is the evidence?\n"
        "Every suggestion below comes from a rule mined from real Instacart orders. "
        "The rule, and the numbers behind it, are shown rather than hidden."
    )

    reference = engine.reference_row
    search = engine.search or {}
    latency = recommendation.elapsed_ms

    a, b, c, d = st.columns(4)
    a.metric(
        "Rules serving now", f"{engine.n_rules:,}",
        help="Association rules in the production rule set the app is answering from.",
    )
    b.metric(
        "Reference top lift",
        f"{reference['top_lift']:.1f}x" if reference and reference.get("n_rules") else "—",
        help=(
            "Highest lift found by the external reference library (mlxtend) on the "
            "benchmark pass: same corpus, support ≥ "
            f"{engine.benchmarks.get('params', {}).get('min_support')}, so every rule in "
            "it is backed by at least "
            f"{engine.benchmarks.get('params', {}).get('min_support', 0) * headline.get('n_transactions', 0):,.0f} "
            "orders. Context for the numbers below, not a target and not a "
            "state-of-the-art claim."
        ),
    )
    c.metric(
        "Evolved mean lift",
        f"{search.get('best_mean_lift', 0):.2f}x" if search else "not run yet",
        help=(
            "Best mean lift the exploratory parameter search reached in its last run. "
            "It is NOT the rule set being served: the search's best configuration uses "
            "a much lower support floor (thinner evidence per rule) and is reported, not "
            f"applied. The served rule set's mean lift is {headline.get('mean_lift', 0):.2f}x."
        ),
    )
    d.metric(
        "Suggestion latency",
        f"{latency:.2f} ms",
        help="Measured time for the lookup that produced the suggestions on this page. "
             "Nothing is fitted at request time; this is a dictionary lookup over "
             "precomputed rule metrics.",
    )


def _economics(recommendation) -> None:
    left, right = st.columns(2)
    left.metric(
        "Basket value (illustrative)",
        f"${recommendation.basket_value_illustrative:,.2f}",
        help=glossary.tooltip("illustrative price"),
    )
    right.metric(
        "Potential add-on value (illustrative)",
        f"${recommendation.addon_value_illustrative:,.2f}",
        help=(
            "Sum of the illustrative prices of the top three suggestions. A description "
            "of what is being suggested, not measured or forecast revenue — the prices "
            "it adds up are synthetic."
        ),
    )


# --------------------------------------------------------------------------- #
# Basket
# --------------------------------------------------------------------------- #

def _basket_panel(engine, basket: list[int]) -> None:
    st.subheader("Basket")

    presets = state.default_presets(engine)
    if presets:
        st.caption(
            "Quick start — each of these is the antecedent of a high-lift rule found "
            "in this corpus, not an invented scenario."
        )
        columns = st.columns(len(presets))
        for column, preset in zip(columns, presets):
            label = preset["label"]
            short = label if len(label) <= 28 else label[:26] + "…"
            if column.button(short, key=f"preset-{preset['label']}", width="stretch",
                             help=f"{label}  ·  top lift {preset['top_lift']:.1f}x"):
                state.set_basket(preset["items"])
                st.rerun()

    catalog = engine.catalog_sorted()
    options = [p for p in catalog if p["product_id"] not in basket]
    choice = st.selectbox(
        "Add a product",
        options=options,
        index=None,
        format_func=lambda p: (
            f"{p['product_name']}  ·  {p['department']}  ·  "
            f"${p['price_illustrative']:.2f} (illustrative)"
        ),
        placeholder="Search the catalog…",
        help="Products already in the basket are not listed.",
    )
    if choice is not None:
        state.add_to_basket(choice["product_id"])
        st.rerun()

    st.caption(f"{len(basket)} product(s) in the basket")
    if not basket:
        if presets:
            st.info("The basket is empty. Add a product, or pick one of the quick-start "
                    "baskets above.")
        elif not engine.n_rules:
            st.info("The basket is empty, and the rule set being served is empty too, so "
                    "no suggestions are possible yet. Loosen the thresholds on the admin "
                    "**Re-mine** tab.")
        else:
            st.info("The basket is empty. Add a product from the catalog.")
        return

    for product_id in basket:
        meta = engine.catalog.get(product_id, {})
        row = st.columns([6, 1])
        row[0].markdown(
            f"**{meta.get('product_name', product_id)}**  \n"
            f"<span style='color:#7a8598;font-size:0.85em'>{meta.get('department', '')} · "
            f"${meta.get('price_illustrative', 0):.2f} illustrative</span>",
            unsafe_allow_html=True,
        )
        if row[1].button("✕", key=f"rm-{product_id}", help="Remove from basket"):
            state.remove_from_basket(product_id)
            st.rerun()


# --------------------------------------------------------------------------- #
# Recommendations
# --------------------------------------------------------------------------- #

def _recommendations(engine, recommendation) -> None:
    st.subheader("Suggested add-ons")

    if recommendation.basket_size == 0:
        st.info("Add something to the basket to see suggestions.")
        return

    if recommendation.is_empty:
        st.warning(
            "No rule fired for this basket. Nothing in the mined rule set predicts an "
            "add-on from these products together — which is a real answer, not an error. "
            "Try a quick-start basket, or add a complementary product: rules are densest "
            "around fresh produce, sparkling water and yogurt."
        )
        return

    st.caption(
        f"{recommendation.n_rules_fired} rule(s) fired for this basket · "
        f"looked up in {recommendation.elapsed_ms:.2f} ms"
    )

    for suggestion in recommendation.suggestions:
        with st.container(border=True):
            head, action = st.columns([5, 1])
            head.markdown(
                f"**{suggestion.product_name}**  \n"
                f"<span style='color:#7a8598;font-size:0.85em'>{suggestion.department} · "
                f"${suggestion.price_illustrative:.2f} illustrative</span>",
                unsafe_allow_html=True,
            )
            if action.button("Add", key=f"add-{suggestion.product_id}", width="stretch"):
                state.add_to_basket(suggestion.product_id)
                st.rerun()

            st.markdown(f"_{suggestion.reason}_")

            metrics = st.columns(3)
            metrics[0].metric("Lift", f"{suggestion.lift:.2f}x", help=glossary.tooltip("lift"))
            metrics[1].metric("Confidence", f"{suggestion.confidence:.0%}",
                              help=glossary.tooltip("confidence"))
            metrics[2].metric("Support", f"{suggestion.support * 100:.2f}%",
                              help=glossary.tooltip("support"))

            st.caption(f"Ranked on: {suggestion.headline_rule}")
            if len(suggestion.supporting_rules) > 1:
                with st.expander("Other rules supporting this suggestion"):
                    for rule in suggestion.supporting_rules[1:]:
                        st.caption(
                            f"{rule['text']} — lift {rule['lift']:.2f}x, "
                            f"confidence {rule['confidence']:.0%}"
                        )


# --------------------------------------------------------------------------- #
# Metric strip
# --------------------------------------------------------------------------- #

def _metric_strip() -> None:
    st.subheader("How to read these numbers")
    columns = st.columns(4)
    formulas = {
        "support": "support(X ➔ Y) = baskets containing X and Y ÷ all baskets",
        "confidence": "confidence(X ➔ Y) = baskets with X and Y ÷ baskets with X",
        "lift": "lift(X ➔ Y) = support(X ➔ Y) ÷ (support(X) × support(Y))",
        "conviction": "conviction(X ➔ Y) = (1 − support(Y)) ÷ (1 − confidence)",
    }
    for column, term in zip(columns, ("support", "confidence", "lift", "conviction")):
        with column:
            st.markdown(f"**{term.title()}**")
            st.code(formulas[term], language=None)
            st.caption(glossary.define(term))


# --------------------------------------------------------------------------- #
# Affinity network
# --------------------------------------------------------------------------- #

def isolation(edges: list[dict], node_ids: set[int], selected: int | None,
              visible_limit: int) -> tuple[list[dict], set[int], int | None]:
    """Which edges to draw and which nodes to keep bright.

    Returns (edges to draw, node ids to highlight, effective selection). A
    selection that is no longer in the graph — say, after a re-mine dropped that
    product — is discarded rather than rendered as an isolation of nothing.
    """
    if selected is not None and selected not in node_ids:
        selected = None
    if selected is None:
        drawn = edges[:visible_limit]
        return drawn, set(node_ids), None
    drawn = [e for e in edges if selected in (e["source"], e["target"])]
    highlighted = {selected} | {e["source"] for e in drawn} | {e["target"] for e in drawn}
    return drawn, highlighted, selected


def _affinity_graph(engine) -> None:
    st.subheader("Product affinity network")
    payload = engine.graph
    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])

    if not nodes:
        st.info("No single-product ➔ single-product rules at the current thresholds, so "
                "there is no pairwise network to draw.")
        return

    st.caption(
        f"{len(edges)} strongest product-to-product rules, of "
        f"{payload.get('n_pairwise_rules_available', 0)} available "
        f"(out of {payload.get('n_rules_total', 0)} rules in total). "
        f"{payload.get('edge_semantics', '')} "
        f"Layout: {payload.get('layout', '')}. Click a product to isolate its rules."
    )

    node_ids = {n["product_id"] for n in nodes}
    drawn, highlighted, selected = isolation(
        edges, node_ids, st.session_state.selected_node,
        payload.get("visible_edges", config.GRAPH_VISIBLE_EDGES),
    )
    if selected != st.session_state.selected_node:
        st.session_state.selected_node = selected          # stale selection dropped
    positions = {n["product_id"]: (n["x"], n["y"]) for n in nodes}

    figure = go.Figure()
    for edge in drawn:
        x0, y0 = positions[edge["source"]]
        x1, y1 = positions[edge["target"]]
        figure.add_trace(
            go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                # Width encodes lift, the metric the edges are ranked on.
                line=dict(width=min(1 + edge["lift"] / 4, 9), color="#7a8fb8"),
                opacity=0.85 if selected is not None else 0.45,
                hoverinfo="text",
                hovertext=(
                    f"{edge['text']}<br>lift {edge['lift']:.2f}x · "
                    f"confidence {edge['confidence']:.0%} · support {edge['support']*100:.2f}%"
                ),
                showlegend=False,
            )
        )

    # Tooltip text travels in customdata[1]. It must be referenced from the
    # hovertemplate: Plotly ignores hovertext whenever a hovertemplate is set.
    # customdata[0] stays the product_id, which the click handler reads.
    # The illustrative price is joined from the catalog here, at render time; the
    # graph artifact itself carries no price.
    tooltips = [
        f"{n['department']} · appears in {n['marginal_frequency']:.1%} of baskets<br>"
        f"${engine.catalog.get(n['product_id'], {}).get('price_illustrative', 0):.2f} "
        f"(illustrative price)"
        for n in nodes
    ]
    figure.add_trace(
        go.Scatter(
            x=[n["x"] for n in nodes], y=[n["y"] for n in nodes],
            mode="markers+text",
            marker=dict(
                size=[26 if n["product_id"] == selected else 15 for n in nodes],
                color=[n["color"] for n in nodes],
                line=dict(
                    width=[3 if n["product_id"] == selected else 1 for n in nodes],
                    color="#1f2733",
                ),
                opacity=[1.0 if n["product_id"] in highlighted else 0.25 for n in nodes],
            ),
            text=[n["product_name"] for n in nodes],
            textposition="top center",
            textfont=dict(size=9),
            customdata=[[n["product_id"], tip] for n, tip in zip(nodes, tooltips)],
            hovertemplate="<b>%{text}</b><br>%{customdata[1]}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=560, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, range=[-1.35, 1.35]),
        yaxis=dict(visible=False, range=[-1.25, 1.25], scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
    )

    event = st.plotly_chart(figure, width="stretch", on_select="rerun", key="affinity")
    points = (event or {}).get("selection", {}).get("points", [])
    picked = next((p.get("customdata") for p in points if p.get("customdata") is not None), None)
    if isinstance(picked, list):
        picked = picked[0] if picked else None

    if picked is not None and picked != selected:
        st.session_state.selected_node = int(picked)
        st.rerun()

    if selected is not None:
        meta = engine.catalog.get(selected, {})
        bar = st.columns([5, 1])
        bar[0].info(
            f"Showing the {len(drawn)} rule(s) touching "
            f"**{meta.get('product_name', selected)}** "
            f"({meta.get('department', '')}, appears in "
            f"{meta.get('marginal_frequency_in_corpus', 0):.1%} of baskets)."
        )
        if bar[1].button("Clear", key="clear-node", width="stretch"):
            st.session_state.selected_node = None
            st.rerun()
