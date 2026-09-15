"""Data-science admin console: benchmarks, AutoResearch and persona radar
(F03, F04, F08, F10, F12, S05–S11)."""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from segmentation import config, elbow as elbow_module

from . import state, theme



# --------------------------------------------------------------------------- #
# S05 — KPI tiles, leaderboard, and the external reference panel (F04)
# --------------------------------------------------------------------------- #

def _kpi_tiles(engine) -> None:
    production = engine.production
    if not production:
        st.info("No benchmark artifacts yet — run the pipeline to populate this panel.")
        return

    glossary = config.METRIC_GLOSSARY
    columns = st.columns(4)
    columns[0].metric("Production algorithm", production.get("algorithm", "—"))
    columns[0].caption(production.get("formulation", ""))
    for column, key, value in (
        (columns[1], "silhouette", theme.metric_or_dash(production.get("silhouette"))),
        (columns[2], "davies_bouldin", theme.metric_or_dash(production.get("davies_bouldin"))),
        (columns[3], "calinski_harabasz",
         theme.metric_or_dash(production.get("calinski_harabasz"), "{:,.1f}")),
    ):
        entry = glossary[key]
        column.metric(entry["name"], value, help=f"{entry['plain']}\n\n{entry['reading']}")
        column.caption(f"{entry['short']} {entry['direction']}.")

    with st.expander("📖 What these numbers mean, in plain language"):
        st.caption(
            "Every metric here is *internal*: it judges the shape of the segments themselves, "
            "because the dataset has no ground-truth labels to check the answer against."
        )
        for entry in glossary.values():
            st.markdown(
                f"**{entry['name']}** — {entry['short']}  \n"
                f"{entry['plain']}  \n"
                f"*How to read it:* {entry['reading']}  \n"
                f"*{entry['direction']} · range {entry['range']}*"
            )
            st.markdown("---")


def _leaderboard(engine) -> None:
    rows = engine.leaderboard
    if not rows:
        return

    production = engine.production
    leader = next((r for r in rows if r.get("silhouette") is not None), None)

    table = []
    for rank, row in enumerate(rows, start=1):
        is_production = (
            row["algorithm"] == production.get("algorithm")
            and row["n_clusters"] == production.get("n_clusters")
        )
        if row.get("degenerate"):
            status = "Degenerate"
        elif leader is not None and row is leader:
            status = "Benchmark leader"
        else:
            status = "Benchmarked"
        table.append(
            {
                "#": rank,
                "Algorithm": row["algorithm"],
                "Family": row["family"],
                "Formulation": row["formulation"],
                "Clusters": row["n_clusters"],
                "Silhouette": row["silhouette"],
                "Davies–Bouldin": row["davies_bouldin"],
                "Calinski–Harabasz": row["calinski_harabasz"],
                "Noise": row["noise_ratio"],
                "Fit (s)": row["fit_seconds"],
                "Status": status + (" · served" if is_production else ""),
            }
        )

    st.dataframe(
        pd.DataFrame(table),
        hide_index=True,
        width="stretch",
        column_config={
            "Silhouette": st.column_config.NumberColumn(
                format="%.4f", help=config.METRIC_GLOSSARY["silhouette"]["short"]
            ),
            "Davies–Bouldin": st.column_config.NumberColumn(
                format="%.4f", help=config.METRIC_GLOSSARY["davies_bouldin"]["short"]
            ),
            "Calinski–Harabasz": st.column_config.NumberColumn(
                format="%.1f", help=config.METRIC_GLOSSARY["calinski_harabasz"]["short"]
            ),
            "Noise": st.column_config.NumberColumn(
                format="percent", help=config.METRIC_GLOSSARY["noise_ratio"]["plain"]
            ),
            "Fit (s)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.caption(engine.benchmarks.get("evaluation_note", ""))

    degenerate = [r for r in rows if r.get("degenerate")]
    if degenerate:
        names = ", ".join(r["algorithm"] for r in degenerate)
        st.markdown(
            f"<div class='caveat'><b>{names} produced fewer than two clusters.</b> The validity "
            f"metrics are mathematically undefined for such a partition, so they are reported as "
            f"“—” rather than given a placeholder score that would let a failed run take a rank.</div>",
            unsafe_allow_html=True,
        )

    dbscan = engine.benchmarks.get("dbscan")
    if dbscan:
        with st.expander("How DBSCAN's ε was chosen (no metric tuning)"):
            st.write(dbscan["rule"])
            st.metric("Chosen ε", f"{dbscan['eps']:.3f}")
            curve = go.Figure(
                go.Scatter(x=dbscan["k_distance_index"], y=dbscan["k_distance_curve"],
                           mode="lines", line=dict(color="#0EA5E9", width=2))
            )
            curve.add_hline(y=dbscan["eps"], line_dash="dot", line_color="#F59E0B",
                            annotation_text=f"ε = {dbscan['eps']:.3f}")
            curve.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10),
                                xaxis_title=f"Customers sorted by {dbscan['min_samples']}-NN distance",
                                yaxis_title="Distance",
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(curve, width="stretch")
            st.caption(
                "ε is fixed by the geometry of the data alone. Tuning it on silhouette would give "
                "DBSCAN an optimisation advantage none of the other entrants receive and would make "
                "the ranking meaningless."
            )


def _external_reference() -> None:
    """F04 — contextual external reference, deliberately outside the ranking."""
    st.markdown("#### External reference points")
    st.markdown(f"<div class='caveat'>{config.EXTERNAL_REFERENCE_CAVEAT}</div>", unsafe_allow_html=True)

    for reference in config.EXTERNAL_REFERENCES:
        st.markdown(
            theme.compact(
            f"""<div class='panel'>
              <b>{reference['label']}</b><br>
              <span class='note'>Reported silhouette <b>{reference['silhouette_text']}</b> at k={reference['k']}
              — <i>not comparable to the figures above.</i></span><br>
              <span class='note'>Setup: {reference['setup']}</span><br>
              <span class='note'>Source: <a href="{reference['url']}" target="_blank">{reference['citation']}</a></span>
            </div>""",
            ),
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# S06 — elbow and silhouette-vs-k
# --------------------------------------------------------------------------- #

def _elbow(engine) -> None:
    sweep = engine.elbow
    if not sweep:
        st.info("No elbow sweep available.")
        return

    rows = sweep["rows"]
    ks = [r["k"] for r in rows]
    served, elbow_k, peak_k = sweep["served_k"], sweep["elbow_k"], sweep["silhouette_peak_k"]

    st.markdown("#### How many segments? The k sweep")
    st.caption(
        f"The model was refitted once for every k from {min(ks)} to {max(ks)}, recording two "
        f"different measures each time. **Inertia (WCSS)** is the total distance from customers to "
        f"their own segment centre — it always falls as k rises, so what matters is the *elbow*, "
        f"the point where it stops dropping steeply. **Silhouette** measures how cleanly separated "
        f"the segments are. The two do not have to agree, and here they do not."
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        figure = go.Figure()
        figure.add_trace(go.Scatter(
            x=ks, y=[r["wcss"] for r in rows], mode="lines+markers", name="WCSS",
            line=dict(color="#7C3AED", width=2.5), marker=dict(size=8),
        ))
        # When the elbow and the served k coincide, draw one marker, not two
        # stacked on the same point with contradicting legend entries.
        markers = ({elbow_k: "inertia elbow"} if elbow_k != served
                   else {served: "inertia elbow = served k"})
        markers.setdefault(served, "served k")
        for k, label in markers.items():
            row = next(r for r in rows if r["k"] == k)
            figure.add_trace(go.Scatter(
                x=[k], y=[row["wcss"]], mode="markers", name=f"k={k} ({label})",
                marker=dict(size=16, color="#10B981" if k == served else "#F59E0B",
                            line=dict(width=2, color="#fff")),
            ))
        figure.update_layout(
            height=330, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="k", tickmode="array", tickvals=ks),
            yaxis=dict(title="Within-cluster sum of squares"),
            legend=dict(orientation="h", y=1.12, font=dict(size=10)),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, width="stretch")

    with right:
        table = pd.DataFrame(
            [
                {
                    "k": r["k"],
                    "WCSS": r["wcss"],
                    "Silhouette": r["silhouette"],
                    "Notes": ("← served " if r["k"] == served else "")
                             + ("★ silhouette peak " if r["k"] == peak_k else "")
                             + ("⌐ inertia elbow" if r["k"] == elbow_k else ""),
                }
                for r in rows
            ]
        )
        st.dataframe(
            table, hide_index=True, width="stretch", height=330,
            column_config={
                "WCSS": st.column_config.NumberColumn(
                    format="%.0f", help=config.METRIC_GLOSSARY["wcss"]["short"]
                ),
                "Silhouette": st.column_config.NumberColumn(
                    format="%.4f", help=config.METRIC_GLOSSARY["silhouette"]["short"]
                ),
            },
        )

    verdict = elbow_module.verdict(sweep)
    if verdict["support"] == "both":
        st.success(f"{verdict['headline']} {verdict['detail']}")
    else:
        st.markdown(
            f"<div class='caveat'><b>{verdict['headline']}</b> {verdict['detail']}</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# S07–S10 — AutoResearch
# --------------------------------------------------------------------------- #

def _step_modal(step: dict) -> None:
    @st.dialog(f"Step {step['round']} · {step['component']}", width="large")
    def show() -> None:
        decision = step["decision"]
        colour = {"accepted": "#10B981", "rejected": "#EF4444"}.get(decision, "#0EA5E9")
        st.markdown(
            theme.compact(
            f"""<div class='panel' style="border-left:4px solid {colour};">
              <b>{step['phase_label']}</b> · <span class='badge {"acc" if decision == "accepted" else "rej"}'>{decision.upper()}</span><br>
              <span class='note'>before <b>{step['silhouette_before']:.5f}</b> →
              after <b>{step['silhouette_after']:.5f}</b> &nbsp;(Δ {step['delta']:+.5f}) &nbsp;·&nbsp;
              gate {config.AUTORESEARCH_GATE:+.4f} silhouette and ≥{config.MIN_CLUSTER_SHARE:.0%} per cluster</span>
            </div>""",
            ),
            unsafe_allow_html=True,
        )
        st.markdown("**Hypothesis**")
        st.write(step["hypothesis"])
        st.markdown("**Transformation**")
        st.code(step["transformation"], language="python")
        st.markdown("**Reflection**")
        st.write(step["reflection"])
        partition = step.get("partition", {})
        if partition.get("sizes"):
            sizes = sorted(partition["sizes"].values(), reverse=True)
            st.markdown("**Resulting partition**")
            st.write(f"{len(sizes)} clusters, sizes {sizes} "
                     f"(smallest holds {partition['smallest_share']:.1%} of customers)")
        st.markdown("**Active hyperparameters**")
        st.code(json.dumps(step["params"], indent=2), language="json")
        st.markdown("**Active features**")
        st.write(", ".join(step["active_features"]))

    show()


def _autoresearch(engine) -> None:
    record = engine.research
    if not record:
        st.info("No AutoResearch run on record. Launch one below.")
        _autoresearch_launcher(engine)
        return

    # --- backbone grid (S07) ---
    st.markdown("#### Phase 1 · backbone tournament")
    st.caption(
        f"Five families evaluated on the {record['n_rows']:,}-customer search sample using the "
        f"eight raw attributes only — the engineered features are what the search goes on to rediscover."
    )
    columns = st.columns(len(record["backbone"]))
    best = max((b for b in record["backbone"] if b["silhouette"] is not None),
               key=lambda b: b["silhouette"], default=None)
    for column, entry in zip(columns, record["backbone"]):
        champion = best is not None and entry["algorithm"] == best["algorithm"]
        column.markdown(
            theme.compact(
            f"""<div class='persona-card' style="border-left-color:{'#10B981' if champion else '#64748B'};">
              <div class='idx'>{entry['family']}</div>
              <div class='nm' style='font-size:.95rem'>{entry['algorithm']}</div>
              <div class='stat'><span>Silhouette</span><b>{theme.metric_or_dash(entry['silhouette'])}</b></div>
              <div class='stat'><span>Davies–Bouldin</span><b>{theme.metric_or_dash(entry['davies_bouldin'])}</b></div>
              <div class='stat'><span>Calinski–H.</span><b>{theme.metric_or_dash(entry['calinski_harabasz'], '{:,.0f}')}</b></div>
              <div class='stat'><span>Fit</span><b>{entry['fit_seconds']:.2f}s</b></div>
              {"<div class='badge lead' style='margin-top:.4rem'>CHAMPION</div>" if champion else ""}
            </div>""",
            ),
            unsafe_allow_html=True,
        )

    # --- KPI tiles ---
    st.markdown("#### Search outcome")
    columns = st.columns(4)
    columns[0].metric("Starting silhouette", f"{record['start_silhouette']:.5f}")
    columns[0].caption("Best backbone family, raw attributes")
    columns[1].metric("Best silhouette", f"{record['best_silhouette']:.5f}",
                      f"{record['improvement_pct']:+.1f}%")
    columns[1].caption("vs this run's own starting score")
    columns[2].metric("Total steps", record["total_steps"])
    columns[3].metric("Accepted / rejected", f"{record['accepted']} / {record['rejected']}")

    st.markdown(
        f"<div class='caveat'><b>What this number is and is not.</b> "
        f"{record['improvement_basis']} {record['promotion_note']}</div>",
        unsafe_allow_html=True,
    )
    with st.expander("Acceptance gate"):
        st.write(record.get("acceptance_gate", ""))

    # --- S09 filters ---
    phases = ["All phases"] + [step["phase_label"] for step in record["steps"]]
    phases = list(dict.fromkeys(phases))
    filters = st.columns([2, 2, 3])
    phase_filter = filters[0].selectbox("Phase", phases, key="phase_filter")
    decision_filter = filters[1].selectbox(
        "Decision", ["All decisions", "accepted", "rejected", "baseline"], key="decision_filter"
    )

    steps = [
        step for step in record["steps"]
        if (phase_filter == "All phases" or step["phase_label"] == phase_filter)
        and (decision_filter == "All decisions" or step["decision"] == decision_filter)
    ]

    # --- trajectory chart (S07) ---
    st.markdown("#### Optimisation trajectory")
    figure = go.Figure()
    all_steps = record["steps"]
    figure.add_trace(go.Scatter(
        x=[s["round"] for s in all_steps], y=[s["silhouette_after"] for s in all_steps],
        mode="lines", line=dict(color="rgba(148,163,184,.55)", width=1.5), showlegend=False,
        hoverinfo="skip",
    ))
    for decision, colour in (("accepted", "#10B981"), ("rejected", "#EF4444"), ("baseline", "#0EA5E9")):
        subset = [s for s in all_steps if s["decision"] == decision]
        if not subset:
            continue
        figure.add_trace(go.Scatter(
            x=[s["round"] for s in subset], y=[s["silhouette_after"] for s in subset],
            mode="markers+text", name=decision.title(),
            text=[str(s["round"]) for s in subset], textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=13, color=colour, line=dict(width=1.5, color="#fff")),
            customdata=[[s["component"], s["delta"]] for s in subset],
            hovertemplate="<b>%{customdata[0]}</b><br>silhouette %{y:.5f}<br>Δ %{customdata[1]:+.5f}<extra></extra>",
        ))
    figure.update_layout(
        height=330, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Step", dtick=1), yaxis=dict(title="Silhouette after step"),
        legend=dict(orientation="h", y=1.14, font=dict(size=10)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")

    # --- experiment stream (S07/S08) ---
    st.markdown(f"#### Experiment stream · {len(steps)} of {len(all_steps)} steps")
    if not steps:
        st.info(
            f"No steps match this combination. “{phase_filter}” and “{decision_filter}” exclude "
            f"each other in this run — try widening one of them."
        )
    for step in steps:
        columns = st.columns([0.6, 1.6, 4.2, 1.1, 1.1, 1.2])
        columns[0].markdown(f"**{step['round']}**")
        columns[1].caption(step["phase_label"])
        columns[2].markdown(f"**{step['component']}** — {step['hypothesis']}")
        columns[3].markdown(f"<span class='mono'>{step['silhouette_after']:.5f}</span>", unsafe_allow_html=True)
        columns[4].markdown(f"<span class='mono'>{step['delta']:+.5f}</span>", unsafe_allow_html=True)
        badge = {"accepted": "acc", "rejected": "rej"}.get(step["decision"], "plain")
        with columns[5]:
            st.markdown(f"<span class='badge {badge}'>{step['decision'].upper()}</span>", unsafe_allow_html=True)
            if st.button("Inspect", key=f"inspect-{step['round']}", width="stretch"):
                _step_modal(step)

    # --- S10 export + F12 relaunch ---
    st.divider()
    columns = st.columns([2, 2, 4])
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    columns[0].download_button(
        "⬇ Export run record (JSON)",
        data=json.dumps(record, indent=2),
        file_name=f"autoresearch-run-{stamp}.json",
        mime="application/json",
        width="stretch",
    )
    with columns[1]:
        _autoresearch_launcher(engine, compact=True)


def _autoresearch_launcher(engine, compact: bool = False) -> None:
    """F12 — launch a fresh run, replacing the stored record."""
    with st.popover("🔬 Run new search", width="stretch") if compact else st.container():
        rows = st.slider("Search sample (customers)", 500,
                         int(engine.meta.get("n_available", config.AUTORESEARCH_SAMPLE)),
                         min(config.AUTORESEARCH_SAMPLE, int(engine.meta.get("n_available", 1600))),
                         step=100, key="ar-rows")
        k = st.number_input("Clusters", config.MIN_K, config.MAX_K,
                            int(engine.meta.get("k", config.DEFAULT_K)), key="ar-k")
        seed = st.number_input("Seed", 0, 10_000, config.DEFAULT_SEED, key="ar-seed")
        if st.button("Launch AutoResearch", type="primary", width="stretch", key="ar-go"):
            try:
                with st.spinner("Running four-phase search…"):
                    state.run_autoresearch(n_rows=int(rows), k=int(k), seed=int(seed))
                st.session_state.action_message = ("success", "AutoResearch complete — run record replaced.")
                st.balloons()
                st.rerun()
            except Exception as exc:
                st.error(f"AutoResearch failed: {exc}")


# --------------------------------------------------------------------------- #
# S11 — persona radar
# --------------------------------------------------------------------------- #

def _radar(engine) -> None:
    personas = engine.personas
    if not personas:
        st.info("No persona profiles available.")
        return

    options = ["All clusters"] + [p["name"] for p in personas]
    focus = st.radio("Focus", options, horizontal=True, key="radar_focus", label_visibility="collapsed")

    axes = config.RADAR_AXES
    labels = [config.ATTRIBUTE_LABELS[a] for a in axes]

    left, right = st.columns([3, 2], gap="large")

    with left:
        figure = go.Figure()
        for record in personas:
            isolated = focus != "All clusters" and record["name"] == focus
            dimmed = focus != "All clusters" and not isolated
            if dimmed:
                continue
            values = [
                min(1.0, max(config.RADAR_FLOOR, record["means"][axis] / config.RADAR_MAXIMA[axis]))
                for axis in axes
            ]
            figure.add_trace(go.Scatterpolar(
                r=values + [values[0]], theta=labels + [labels[0]],
                name=record["name"], fill="toself",
                line=dict(color=record["color"], width=3 if isolated else 1.8),
                opacity=0.85 if isolated else 0.45,
            ))
        figure.update_layout(
            height=430, margin=dict(l=40, r=40, t=30, b=30),
            polar=dict(radialaxis=dict(range=[0, 1], showticklabels=True, tickfont=dict(size=9))),
            legend=dict(orientation="h", y=-0.08, font=dict(size=10)),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "Each axis is divided by a fixed display maximum ("
            + ", ".join(f"{config.ATTRIBUTE_LABELS[a]} {config.RADAR_MAXIMA[a]:g}" for a in axes)
            + f") and clamped to [{config.RADAR_FLOOR}, 1], so shapes are comparable across personas."
        )

    with right:
        for record in personas:
            dimmed = focus != "All clusters" and record["name"] != focus
            means = record["means"]
            st.markdown(
                theme.compact(
                f"""<div class='persona-card' style="border-left-color:{record['color']};
                     opacity:{.35 if dimmed else 1};margin-bottom:.5rem;">
                  <div class='nm' style='font-size:.92rem'>{record['badge']} {record['name']}</div>
                  <div class='stat'><span>Income</span><b>{theme.money(means['income_k'] * 1000)}</b></div>
                  <div class='stat'><span>Spending score</span><b>{means['spending_score']:.0f}/100</b></div>
                  <div class='stat'><span>Annual spend</span><b>{theme.money(means['total_spend'])}</b></div>
                  <div class='stat'><span>Recency</span><b>{means['recency_days']:.0f} days</b></div>
                  <div class='stat'><span>Persona fit</span><b>{f"{record['match_quality']:.0%}" if record.get('matched', True) else 'none — data-derived'}</b></div>
                </div>"""
                ),
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #

def render(engine) -> None:
    header = st.columns([4, 1])
    header[0].markdown("### Data-science console")
    if header[1].button("↻ Refresh telemetry", width="stretch"):
        state.invalidate()
        st.rerun()

    st.segmented_control(
        "Section", state.ADMIN_TABS, key="admin_tab_control",
        label_visibility="collapsed",
    )
    tab = state.current_admin_tab()

    if tab == "Benchmarks":
        _kpi_tiles(engine)
        st.markdown("#### Algorithm leaderboard")
        _leaderboard(engine)
        st.divider()
        _elbow(engine)
        st.divider()
        _external_reference()
    elif tab == "AutoResearch":
        _autoresearch(engine)
    else:
        _radar(engine)
