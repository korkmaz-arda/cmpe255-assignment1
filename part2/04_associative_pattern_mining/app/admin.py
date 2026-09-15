"""Data-science admin workspace: benchmark, parameter search, rules explorer."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import glossary, state
from basket import config

TABS = ("Algorithm benchmark", "Parameter search", "Rules explorer", "Re-mine")


def _metric_or_dash(value: float, n_rules: int, fmt: str) -> str:
    """Lift and confidence are undefined over an empty rule set; show that, not a zero."""
    return format(value, fmt) if n_rules else "—"


def _request(flag: str) -> None:
    """Button callback. Runs before the rerun, so the trigger renders disabled."""
    st.session_state[flag] = True


def show_action_message() -> None:
    """Report the outcome of the last long-running action.

    The action itself ends in `st.rerun()` so every view re-reads the new
    artifacts; anything rendered before that rerun is discarded. The outcome is
    therefore parked in session state and shown here, on the run after.
    """
    message = st.session_state.get("action_message")
    if not message:
        return
    kind, text = message
    if kind == "success":
        st.success(text)
        st.balloons()
    else:
        st.warning(text)
    st.session_state.action_message = None
TAB_ICONS = {
    "Algorithm benchmark": "⚖️",
    "Parameter search": "🧪",
    "Rules explorer": "🔎",
    "Re-mine": "⚙️",
}


def render(engine) -> None:
    # Pills rather than a second segmented bar, so the sub-level reads as
    # subordinate to the main view switch above it.
    tab = st.pills(
        "Admin section", TABS,
        key="admin_tab",
        required=True,
        format_func=lambda option: f"{TAB_ICONS[option]}  {option}",
        label_visibility="collapsed",
    ) or TABS[0]
    st.divider()
    if tab == "Algorithm benchmark":
        _benchmark(engine)
    elif tab == "Parameter search":
        _search(engine)
    elif tab == "Rules explorer":
        _explorer(engine)
    else:
        _remine(engine)


# --------------------------------------------------------------------------- #
# F07 — benchmark
# --------------------------------------------------------------------------- #

def _benchmark(engine) -> None:
    summary = engine.benchmarks
    headline = engine.headline
    params = summary.get("params", {})

    st.subheader("Algorithm benchmark")

    # The tiles describe the production rule set the app serves; the table below
    # describes the benchmark pass. They use different thresholds, so each is
    # labelled with its own rather than sharing one caption.
    production = summary.get("production_params", {})
    n_rules = headline.get("n_active_rules", 0)
    st.markdown("##### Production rule set (what the app serves)")
    st.caption(
        f"support ≥ {production.get('min_support')}, itemsets up to "
        f"{production.get('max_len')} products, confidence ≥ "
        f"{production.get('min_confidence')}, lift ≥ {production.get('min_lift')}, "
        f"mined by {headline.get('production_algorithm')}."
    )
    tiles = st.columns(4)
    tiles[0].metric("Orders mined", f"{headline.get('n_transactions', 0):,}",
                    help="Every sampled order counts, including those with fewer than two "
                         "catalog products — they are the negative evidence in every support.")
    tiles[1].metric("Frequent itemsets", f"{headline.get('n_frequent_itemsets', 0):,}",
                    help=glossary.tooltip("frequent itemset"))
    tiles[2].metric("Active rules", f"{n_rules:,}",
                    help=glossary.tooltip("association rule"))
    tiles[3].metric("Top lift", _metric_or_dash(headline.get("top_lift", 0), n_rules, ".2f")
                    + ("x" if n_rules else ""),
                    help=glossary.tooltip("lift"))

    if n_rules and engine.rule_records:
        top = engine.rule_records[0]
        orders = top["support"] * headline.get("n_transactions", 0)
        st.caption(
            f"The top-lift rule, {top['text']}, is backed by about **{orders:,.0f} orders** "
            f"(support {top['support'] * 100:.3f}%, confidence {top['confidence']:.0%}). "
            f"A very high lift on a rare combination is a strong association, not "
            f"automatically strong business evidence — check the order count behind it."
        )
    elif not n_rules:
        st.warning("No rules survived the production thresholds. Loosen them on the "
                   "**Re-mine** tab.")

    st.markdown("##### Benchmark pass (how the algorithms compare)")
    st.caption(
        f"All entrants mined the same {summary.get('n_transactions', 0):,}-order corpus at the "
        f"same thresholds (support ≥ {params.get('min_support')}, itemsets up to "
        f"{params.get('max_len')} products, confidence ≥ {params.get('min_confidence')}, "
        f"lift ≥ {params.get('min_lift')}). Every figure in this table comes from that one "
        f"pass, and every runtime is the measured time of the run that produced its own row."
    )

    agreement = summary.get("implementations_agree", {})
    if agreement.get("agrees"):
        st.success(
            "The three implementations found identical itemsets at identical supports, as "
            "exact algorithms must. Only their runtimes differ."
        )
    else:
        st.error(f"Implementations disagree: {agreement.get('detail')}")

    rows = []
    for row in summary.get("leaderboard", []):
        rows.append({
            "": "🏆" if row.get("is_champion") else ("🔍" if row.get("is_reference") else ""),
            "Algorithm": row["algorithm"],
            "Role": "External reference" if row.get("is_reference") else "Implemented here",
            "Itemsets": row["n_itemsets"],
            "Rules": row["n_rules"],
            "Top lift": _metric_or_dash(row["top_lift"], row["n_rules"], ".2f"),
            "Mean confidence": _metric_or_dash(row["mean_confidence"], row["n_rules"], ".1%"),
            "Runtime (s)": round(row["elapsed_seconds"], 3),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    champion = summary.get("champion")
    st.caption(
        f"🏆 **{champion}** is the production champion: the fastest of the three "
        f"implementations, on measured time. 🔍 marks the external reference row."
    )

    reference = engine.reference_row
    if reference:
        with st.container(border=True):
            st.markdown(f"**External reference — {reference['algorithm']}**")
            st.caption(reference["paradigm"])
            st.caption(reference.get("comparability_note", ""))
            if reference.get("agrees_with_implementations"):
                st.caption(
                    "✅ It found the same frequent itemsets as the implementations above, "
                    "which is what it is here for."
                )

    st.markdown("##### How the three approaches differ")
    cards = st.columns(3)
    for card, row in zip(cards, [r for r in summary.get("leaderboard", []) if not r["is_reference"]]):
        with card:
            with st.container(border=True):
                st.markdown(f"**{row['algorithm']}**")
                st.caption(row["paradigm"])
                st.caption(f"Memory: {row['memory_note']}")
                st.caption(f"Measured: {row['elapsed_seconds']:.3f}s")


# --------------------------------------------------------------------------- #
# F08 / S08-S11 — parameter search
# --------------------------------------------------------------------------- #

def _search(engine) -> None:
    st.subheader("Automated parameter search")
    record = engine.search

    top = st.columns([3, 1])
    top[0].caption(
        "A four-phase hill climb over mining thresholds. Objective: mean lift across the "
        "surviving rules. Every trial below was actually run; the rejected ones are kept, "
        "because the dead ends are how the final configuration was reached."
    )
    running = st.session_state.get("search_running", False)
    top[1].button(
        "Running…" if running else "Run the search",
        width="stretch", disabled=running,
        on_click=_request, args=("search_running",),
    )
    if running:
        try:
            with st.spinner("Running all four phases…"):
                new_record = state.run_search(config.SEARCH_N_ORDERS)
            st.session_state.action_message = (
                "success",
                f"Search complete: mean lift {new_record['initial_mean_lift']:.3f} → "
                f"{new_record['best_mean_lift']:.3f} "
                f"({new_record['accepted']} accepted, {new_record['rejected']} rejected).",
            )
        finally:
            st.session_state["search_running"] = False
        st.rerun()

    show_action_message()

    if not record:
        st.info("The search has not been run yet. Press **Run the search**, or run "
                "`python -m basket.autoresearch` from the terminal.")
        return

    st.caption(f"Acceptance rule: {record['acceptance_rule']}.")

    best = record["best_params"]
    production = engine.benchmarks.get("production_params", config.PRODUCTION)
    n_search = record["corpus"]["n_transactions"]
    st.info(
        f"**Exploratory, not applied.** The search reports its best configuration — "
        f"support ≥ {best['min_support']} (about "
        f"{best['min_support'] * n_search:,.0f} orders behind an itemset), itemsets up to "
        f"{best['max_len']} products, confidence ≥ {best['min_confidence']}, lift ≥ "
        f"{best['min_lift']} — but does not change what the app serves. The production "
        f"rule set stays at support ≥ {production.get('min_support')} (about "
        f"{production.get('min_support', 0) * engine.headline.get('n_transactions', 0):,.0f} "
        f"orders), because a higher mean lift bought with thinner evidence per rule is "
        f"not automatically a better rule set to act on."
    )

    kpis = st.columns(5)
    kpis[0].metric("Starting mean lift", f"{record['initial_mean_lift']:.3f}",
                   help=glossary.tooltip("mean lift"))
    kpis[1].metric("Best mean lift", f"{record['best_mean_lift']:.3f}",
                   delta=f"{record['best_mean_lift'] - record['initial_mean_lift']:+.3f}")
    kpis[2].metric("Improvement", f"{record['improvement_pct']:+.1f}%")
    kpis[3].metric("Accepted", record["accepted"])
    kpis[4].metric("Rejected", record["rejected"])

    _tournament_cards(record)
    _trajectory_chart(record)
    _trajectory_table(record)


def _tournament_cards(record: dict) -> None:
    st.markdown("##### Phase 1 — backbone tournament")
    st.caption(
        "The three algorithms at the baseline thresholds. They are exact, so they agree on "
        "the rule set; the tournament is choosing a runtime, not a result."
    )
    cards = st.columns(len(record["tournament"]))
    for card, row in zip(cards, record["tournament"]):
        with card, st.container(border=True):
            badge = " 🏆" if row.get("is_champion") else ""
            st.markdown(f"**{row['algorithm']}**{badge}")
            st.caption(glossary.define(row["algorithm"]))
            st.metric("Runtime", f"{row['elapsed_seconds']:.3f}s")
            st.caption(f"{row['n_itemsets']:,} itemsets · mean lift {row['mean_lift']:.3f}")


def _trajectory_chart(record: dict) -> None:
    st.markdown("##### Search trajectory")
    steps = record["trajectory"]
    lifts = [s["mean_lift_after"] for s in steps]

    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=[s["iteration"] for s in steps], y=lifts,
        mode="lines", line=dict(color="#7a8598", width=1, dash="dot"),
        hoverinfo="skip", showlegend=False,
    ))
    palette = {"accepted": "#27ae60", "rejected": "#eb5757", "baseline": "#4c9be8"}
    for decision in ("baseline", "accepted", "rejected"):
        subset = [s for s in steps if s["decision"] == decision]
        if not subset:
            continue
        figure.add_trace(go.Scatter(
            x=[s["iteration"] for s in subset],
            y=[s["mean_lift_after"] for s in subset],
            mode="markers", name=decision.title(),
            marker=dict(size=13, color=palette[decision],
                        line=dict(width=1, color="#1f2733")),
            customdata=[s["step_id"] for s in subset],
            hovertext=[
                f"{s['phase']}<br>{s['category']}<br>mean lift {s['mean_lift_after']:.3f} "
                f"({s['delta']:+.3f})<br>{s['n_rules']} rules"
                for s in subset
            ],
            hoverinfo="text",
        ))

    # Axis range follows the data, so no trial can land off-chart.
    low, high = min(lifts), max(lifts)
    pad = max((high - low) * 0.15, 0.05)
    figure.update_layout(
        height=380, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Trial", yaxis_title="Mean lift",
        yaxis=dict(range=[low - pad, high + pad]),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.1),
    )
    event = st.plotly_chart(figure, width="stretch", on_select="rerun", key="trajectory")
    points = (event or {}).get("selection", {}).get("points", [])
    picked = next((p.get("customdata") for p in points if p.get("customdata") is not None), None)
    if isinstance(picked, list):
        picked = picked[0] if picked else None
    if picked is not None:
        _inspect_step(next(s for s in steps if s["step_id"] == int(picked)))
    else:
        st.caption("Click any point to inspect that trial.")


@st.dialog("Trial inspector", width="large")
def _inspect_step(step: dict) -> None:
    st.markdown(f"**{step['phase']} — {step['category']}**")
    badge = {"accepted": "✅ Accepted", "rejected": "❌ Rejected", "baseline": "◻️ Baseline"}
    columns = st.columns(3)
    columns[0].metric("Decision", badge.get(step["decision"], step["decision"]))
    columns[1].metric(
        "Mean lift",
        f"{step['mean_lift_after']:.3f}",
        delta=f"{step['delta']:+.3f}",
    )
    columns[2].metric("Rules surviving", f"{step['n_rules']:,}")

    st.markdown("**Hypothesis**")
    st.write(step["hypothesis"])
    st.markdown("**What happened**")
    st.write(step["reflection"])
    st.markdown("**The configuration tried**")
    st.json(step["params"])
    st.markdown("**The change, in code**")
    st.code(step["code_change"], language="python")
    st.caption(
        f"Trial {step['iteration']} · {step['n_itemsets']:,} itemsets · "
        f"top lift {step['top_lift']:.2f}x · longest antecedent "
        f"{step['max_antecedent_len']} · {step['timestamp']}"
    )


def _trajectory_table(record: dict) -> None:
    st.markdown("##### All trials")
    steps = record["trajectory"]
    phases = ["All phases"] + sorted({s["phase"] for s in steps})
    decisions = ["All decisions", "accepted", "rejected", "baseline"]

    columns = st.columns(2)
    phase = columns[0].selectbox("Phase", phases, key="phase_filter")
    decision = columns[1].selectbox("Decision", decisions, key="decision_filter")

    filtered = [
        s for s in steps
        if (phase == "All phases" or s["phase"] == phase)
        and (decision == "All decisions" or s["decision"] == decision)
    ]
    if not filtered:
        st.info("No trials match these filters.")
        return

    st.dataframe(
        pd.DataFrame([
            {
                "#": s["step_id"],
                "Phase": s["phase"].split("—")[0].strip(),
                "Change": s["category"],
                "Mean lift before": round(s["mean_lift_before"], 3),
                "Mean lift after": round(s["mean_lift_after"], 3),
                "Delta": round(s["delta"], 3),
                "Rules": s["n_rules"],
                "Decision": s["decision"],
            }
            for s in filtered
        ]),
        width="stretch", hide_index=True,
    )


# --------------------------------------------------------------------------- #
# F09 — rules explorer
# --------------------------------------------------------------------------- #

SORTABLE = {
    "support": "support", "confidence": "confidence", "lift": "lift",
    "leverage": "leverage", "conviction": "conviction",
}


def _explorer(engine) -> None:
    st.subheader("Association rules")
    st.caption(
        f"The {config.RULES_TABLE_FEED} strongest rules of the {engine.n_rules:,} in the "
        f"production set, showing up to {config.RULES_TABLE_ROWS}. Search either side of a "
        f"rule, and sort on any metric."
    )

    controls = st.columns([3, 2, 1])
    query = controls[0].text_input("Search products", key="rules_query",
                                   placeholder="e.g. avocado").strip().lower()
    sort_key = controls[1].selectbox("Sort by", list(SORTABLE), key="rules_sort",
                                     help=glossary.tooltip("lift"))
    descending = controls[2].toggle("Descending", key="rules_descending")

    records = engine.top_rules(config.RULES_TABLE_FEED)
    if query:
        records = [
            r for r in records
            if query in " ".join(r["antecedent_names"] + r["consequent_names"]).lower()
        ]
    records = sorted(records, key=lambda r: r[SORTABLE[sort_key]], reverse=descending)
    records = records[: config.RULES_TABLE_ROWS]

    if not records:
        if not engine.n_rules:
            # The cause is the rule set, not the search box.
            st.warning("The production rule set is empty: no rule survived the current "
                       "thresholds. Loosen them on the **Re-mine** tab.")
        else:
            st.info(f"No rules mention “{query}”. Try another product.")
        return

    st.dataframe(
        pd.DataFrame([
            {
                "#": index,
                "If the basket has": " + ".join(r["antecedent_names"]),
                "…it also tends to have": " + ".join(r["consequent_names"]),
                "Support": f"{r['support'] * 100:.2f}%",
                "Confidence": f"{r['confidence'] * 100:.1f}%",
                "Lift": f"{r['lift']:.2f}x",
                "Leverage": f"{r['leverage']:.5f}",
                "Conviction": f"{r['conviction']:.2f}",
            }
            for index, r in enumerate(records, start=1)
        ]),
        width="stretch", hide_index=True,
        column_config={
            "Support": st.column_config.TextColumn(help=glossary.define("support")),
            "Confidence": st.column_config.TextColumn(help=glossary.define("confidence")),
            "Lift": st.column_config.TextColumn(help=glossary.define("lift")),
            "Leverage": st.column_config.TextColumn(help=glossary.define("leverage")),
            "Conviction": st.column_config.TextColumn(help=glossary.define("conviction")),
        },
    )

    with st.expander("What these columns mean"):
        for term in glossary.METRIC_TERMS:
            st.markdown(f"**{term.title()}** — {glossary.define(term)}")


# --------------------------------------------------------------------------- #
# F10 — re-mining
# --------------------------------------------------------------------------- #

def _remine(engine) -> None:
    st.subheader("Re-mine the rule set")
    show_action_message()
    st.caption(
        "Re-runs the whole pipeline and swaps the new artifacts into the running app. "
        "All four controls change the result — nothing here is decorative."
    )

    production = engine.benchmarks.get("production_params", config.PRODUCTION)
    columns = st.columns(2)
    min_support = columns[0].slider(
        "Minimum support", *config.SUPPORT_RANGE,
        value=float(production.get("min_support", config.PRODUCTION["min_support"])),
        step=0.0001, format="%.4f", help=glossary.tooltip("support"),
    )
    min_confidence = columns[1].slider(
        "Minimum confidence", *config.CONFIDENCE_RANGE,
        value=float(production.get("min_confidence", config.PRODUCTION["min_confidence"])),
        step=0.01, help=glossary.tooltip("confidence"),
    )
    min_lift = columns[0].slider(
        "Minimum lift", *config.LIFT_RANGE,
        value=float(production.get("min_lift", config.PRODUCTION["min_lift"])),
        step=0.05, help=glossary.tooltip("lift"),
    )
    n_orders = columns[1].slider(
        "Orders to mine", *config.ORDERS_RANGE,
        value=int(engine.headline.get("n_transactions", config.DEFAULT_N_ORDERS)),
        step=1000,
        help="A seeded random sample of the train split. Fewer orders means less evidence "
             "behind every rule.",
    )

    support_orders = min_support * n_orders
    st.caption(
        f"At these settings an itemset needs about **{support_orders:.0f} orders** behind it "
        f"to be reported."
    )
    if support_orders < 20:
        st.warning(
            f"That is thin evidence: roughly {support_orders:.0f} orders per itemset. The rules "
            f"will still be computed honestly, but expect noise."
        )

    running = st.session_state.get("remining", False)
    st.button(
        "Mining…" if running else "Re-mine now",
        type="primary", disabled=running,
        on_click=_request, args=("remining",),
    )
    if running:
        try:
            with st.spinner("Mining…"):
                meta = state.remine(min_support, min_confidence, min_lift, n_orders)
            headline = meta["headline"]
            if headline["n_active_rules"]:
                st.session_state.action_message = (
                    "success",
                    f"Re-mined: **{headline['n_active_rules']:,} active rules** from "
                    f"{headline['n_frequent_itemsets']:,} frequent itemsets over "
                    f"{headline['n_transactions']:,} orders "
                    f"(top lift {headline['top_lift']:.2f}x, "
                    f"{headline['basket_coverage']:.0%} basket coverage). Every view now "
                    f"reads the new rule set.",
                )
            else:
                st.session_state.action_message = (
                    "warning",
                    f"Re-mined over {headline['n_transactions']:,} orders, but **no rule "
                    f"survived** these thresholds ({headline['n_frequent_itemsets']:,} "
                    f"frequent itemsets were found; none cleared the confidence and lift "
                    f"gates). The app now serves an empty rule set, so no suggestions will "
                    f"appear until you loosen the thresholds and re-mine.",
                )
        finally:
            st.session_state["remining"] = False
        st.rerun()
