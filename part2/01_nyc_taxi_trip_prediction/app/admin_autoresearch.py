"""Admin: AutoResearch leaderboard, KPIs, trajectory, filters, step inspection, export, launch (F12, F14, S06-S08)."""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from taxi.autoresearch import run_session

from app import state
from app.dialogs import step_dialog
from app.ui_common import GLOSSARY, GOOD, CRITICAL, SERIES_1, MUTED, style_fig

PHASES = {"all": "All", "backbone": "Backbone", "feature": "Feature", "hyperparameter": "Hyperparameter",
          "ensemble": "Ensemble"}
DECISIONS = {"all": "All", "ACCEPTED": "Accepted", "REJECTED": "Rejected"}


def _launch_controls() -> None:
    with st.expander("🚀 Run a new AutoResearch session", expanded=False):
        st.markdown("Runs the full four-phase search on a seeded sample of the fit pool and scores every candidate "
                    "on the **same fixed validation split**, so results from different sessions are comparable. "
                    "The test split is never read, and the served model is not changed. Expect a few minutes.")
        with st.form("ar_form"):
            c1, c2 = st.columns(2)
            size = c1.number_input("Training sample size", min_value=10_000, max_value=500_000, value=100_000,
                                   step=10_000)
            seed = c2.number_input("Seed", min_value=0, max_value=10_000, value=42, step=1)
            go_btn = st.form_submit_button("Start session", type="primary",
                                           disabled=not state.trip_store().exists())
        if go_btn:
            with st.status("Running AutoResearch…", expanded=True) as status:
                try:
                    session = run_session(fit_sample_size=int(size), seed=int(seed), store=state.artifact_store(),
                                          trips=state.trip_store(), progress=lambda m: st.write(m))
                except Exception as exc:
                    status.update(label="AutoResearch failed", state="error")
                    st.error(f"Session failed: {exc}")
                    return
                status.update(label=f"Session {session['session_id']} finished: best validation RMSLE "
                                    f"{session['summary']['best_rmsle']:.5f} after {session['summary']['total_steps']} steps",
                              state="complete")
            state.clear_caches()
            st.session_state.pop("ar_last_open", None)
            st.rerun()


def render() -> None:
    session = state.latest_session()
    _launch_controls()
    if session is None:
        st.info("No AutoResearch session has been recorded yet. Start one above, or run "
                "`python scripts/autoresearch.py`.")
        return
    sm, cfg = session["summary"], session["config"]
    st.caption(f"Session `{session['session_id']}` · {cfg['fit_sample_size']:,} training trips (seed {cfg['seed']}) · "
               f"scored on {cfg['validation_rows']:,} fixed validation trips · ε = {cfg['epsilon']:g} · "
               f"runtime {session['runtime_seconds']:.0f} s · test split used: no")

    st.subheader("Backbone leaderboard (validation)")
    lb = session["leaderboard"]
    cols = st.columns(3)
    for i, row in enumerate(lb):
        with cols[i % 3].container(border=True):
            badge = " 🏆 Champion" if row["champion"] else ""
            st.markdown(f"**#{row['rank']} {row['name']}**{badge}")
            st.markdown(f"<span class='mono'>RMSLE {row['rmsle']:.5f} · R² {row['r2']:.3f} · MAE {row['mae_s']:.0f} s<br>"
                        f"fit {row['fit_seconds']:.1f} s · {row['latency_us_per_row']:.2f} µs/trip inference</span>",
                        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Initial RMSLE", f"{sm['initial_rmsle']:.5f}", "XGBoost base configuration", delta_color="off",
              delta_arrow="off")
    c2.metric("Best evolved RMSLE", f"{sm['best_rmsle']:.5f}", f"{-sm['improvement_pct']:.2f}%",
              delta_color="inverse", help="Change relative to the initial XGBoost configuration (lower is better).")
    c3.metric("Total steps", sm["total_steps"], f"{sm['gated_steps']} gated", delta_color="off", delta_arrow="off")
    c4.metric("Gate decisions", f"{sm['accepted']} ✓ / {sm['rejected']} ✗", "accepted / rejected",
              delta_color="off", delta_arrow="off", help=GLOSSARY["Hill climbing"])
    st.caption(f"The hill climb mutates the XGBoost configuration and compares every candidate with that same "
               f"configuration's current score (like-for-like). The tournament champion "
               f"({sm['champion_backbone']}) is shown for context. Best configuration found: `{sm['final_model']}` "
               "(validation only; it is not deployed; the served model changes only through Retrain).")

    steps = pd.DataFrame(session["steps"])
    for col in ("incumbent_before", "incumbent_after", "delta", "rmsle_candidate"):
        steps[col] = pd.to_numeric(steps[col], errors="coerce")
    f1, f2 = st.columns(2)
    phase = f1.segmented_control("Phase", list(PHASES), format_func=PHASES.get, default="all", key="ar_phase")
    decision = f2.segmented_control("Decision", list(DECISIONS), format_func=DECISIONS.get, default="all",
                                    key="ar_decision")
    view = steps
    if phase and phase != "all":
        view = view[view["phase"] == phase]
    if decision and decision != "all":
        view = view[view["decision"] == decision]

    _trajectory(steps, view)
    st.subheader(f"Experiment stream · {len(view)} of {len(steps)} steps")
    table = view[["iteration", "phase", "component", "rmsle_candidate", "incumbent_before", "delta", "decision",
                  "hypothesis"]].rename(columns={"iteration": "#", "phase": "Phase", "component": "Change",
                                                 "rmsle_candidate": "Candidate RMSLE",
                                                 "incumbent_before": "Incumbent before", "delta": "Improvement",
                                                 "decision": "Decision", "hypothesis": "Hypothesis"})
    sel = st.dataframe(table.style.format({"Candidate RMSLE": "{:.5f}", "Incumbent before": "{:.5f}",
                                           "Improvement": "{:+.5f}"}, na_rep="—"),
                       hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                       key=f"ar_table_{phase}_{decision}")
    picked = None
    if sel and sel.selection.rows:
        picked = int(table.iloc[sel.selection.rows[0]]["#"])
    chart_pick = st.session_state.get("ar_chart_pick")
    if chart_pick is not None:
        picked = chart_pick
        st.session_state["ar_chart_pick"] = None
    if picked is not None and st.session_state.get("ar_last_open") != (session["session_id"], picked, phase, decision):
        st.session_state["ar_last_open"] = (session["session_id"], picked, phase, decision)
        step_dialog(next(s for s in session["steps"] if s["iteration"] == picked))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button("⬇️ Export session JSON", json.dumps(session, indent=2),
                       f"autoresearch_{session['session_id']}_{stamp}.json", "application/json")


def _trajectory(steps: pd.DataFrame, view: pd.DataFrame) -> None:
    fig = go.Figure()
    gated = steps[steps["decision"].isin(["ACCEPTED", "REJECTED"])]
    if len(gated):
        fig.add_trace(go.Scatter(x=gated["iteration"], y=gated["incumbent_after"], mode="lines", name="Incumbent",
                                 line=dict(color=MUTED, width=2, shape="hv"), hoverinfo="skip"))
    styles = {"BENCHMARK": ("Backbone benchmark", SERIES_1, 10, "diamond"),
              "ACCEPTED": ("Accepted", GOOD, 14, "circle"),
              "REJECTED": ("Rejected", CRITICAL, 8, "x")}
    for dec, (name, colour, size, symbol) in styles.items():
        part = view[view["decision"] == dec]
        if not len(part):
            continue
        fig.add_trace(go.Scatter(
            x=part["iteration"], y=part["rmsle_candidate"], mode="markers", name=name,
            marker=dict(color=colour, size=size, symbol=symbol, line=dict(color="#1a1a19", width=2)),
            customdata=part[["component", "phase"]].to_numpy(),
            hovertemplate="#%{x} · %{customdata[1]}<br><b>%{customdata[0]}</b><br>RMSLE %{y:.5f}<extra>"
                          + name + "</extra>"))
    style_fig(fig, height=380, title="RMSLE trajectory (validation) · click a point to inspect")
    fig.update_xaxes(title="Step")
    fig.update_yaxes(title="Validation RMSLE")
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="ar_chart")
    points = (event or {}).get("selection", {}).get("points", []) if isinstance(event, dict) else \
        getattr(getattr(event, "selection", None), "points", [])
    if points:
        it = int(points[0]["x"])
        if st.session_state.get("ar_chart_last") != it:
            st.session_state["ar_chart_last"] = it
            st.session_state["ar_chart_pick"] = it
    st.caption("Grey step line = incumbent score after each gated step. Backbone models do not pass through the "
               "gate; the y-axis focuses on the observed range, so Ridge may sit far above the others.")
