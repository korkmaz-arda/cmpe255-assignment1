"""Admin: KPI tiles, benchmark matrix, importances, learning curves, distributions (S03, F10, F11)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from taxi import config
from taxi.features import FEATURE_DESCRIPTIONS

from app.ui_common import GLOSSARY, GOOD, CRITICAL, WARNING, SERIES_1, SERIES_2, SERIES_3, MUTED, style_fig


def kpi_tiles(bundle) -> None:
    md = bundle.metadata
    vm = md["validation_metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active algorithm", md["active_model"], md["version"], delta_color="off", delta_arrow="off")
    c2.metric("Validation RMSLE", f"{vm['rmsle']:.4f}", help=GLOSSARY["RMSLE"])
    c3.metric("Validation R²", f"{vm['r2']:.3f}", help=GLOSSARY["R²"])
    c4.metric("Training corpus", f"{md['rows']['train']:,} trips",
              f"of {md['data_report']['split_sizes']['fit']:,} in fit pool" if md.get("data_report") else None,
              delta_color="off", delta_arrow="off")
    holdout_block(bundle)


def holdout_block(bundle) -> None:
    if bundle.holdout:
        h = bundle.holdout
        m = h["metrics"]
        with st.container(border=True):
            st.markdown(f"**Final holdout evaluation (test split)** · run once on {h['evaluated_at']}"
                        + (" · *forced re-run*" if h.get("forced_rerun") else ""))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Test RMSLE", f"{m['rmsle']:.4f}")
            c2.metric("Test R²", f"{m['r2']:.3f}")
            c3.metric("Test MAE", f"{m['mae_s']:.0f} s")
            c4.metric("Band coverage (test)", f"{h['band_coverage']['coverage']:.1%}",
                      f"nominal {h['band_coverage']['nominal']:.0%}", delta_color="off", delta_arrow="off")
            st.caption("Kept separate from the interactive, validation-based comparisons below. "
                       + GLOSSARY["Final holdout (test)"])
    else:
        st.caption("🔒 **Final holdout: not finalized.** This version has only validation scores. Run "
                   "`python scripts/finalize.py` once, after the configuration is settled, for a one-time test "
                   "evaluation. Retraining never reads the test split.")


def _band(rmsle: float) -> str:
    return "strong" if rmsle < 0.35 else "moderate" if rmsle < 0.45 else "weak"


def benchmark_matrix(bundle) -> None:
    st.subheader("Model benchmark (validation split)")
    status_label = {"active": "✅ Active in production", "reference": "📏 Reference baseline",
                    "benchmark": "Benchmarked"}
    rows = []
    for e in bundle.experiments:
        v = e["validation"]
        hp = {k: val for k, val in e["hyperparameters"].items() if k != "random_state"}
        rows.append({"Model": e["name"], "Family": e["family"],
                     "Hyperparameters": ", ".join(f"{k}={val}" for k, val in hp.items()),
                     "RMSLE": v["rmsle"], "Quality": _band(v["rmsle"]), "R²": v["r2"],
                     "MAE (s)": v["mae_s"], "MAE (min)": round(v["mae_s"] / 60, 1),
                     "Fit time (s)": e["fit_seconds"], "Status": status_label[e["status"]]})
    df = pd.DataFrame(rows)
    colour = {"strong": GOOD, "moderate": WARNING, "weak": CRITICAL}

    def rmsle_style(col):
        return [f"color: {colour[_band(x)]}; font-weight: 600" for x in col]

    styled = df.style.apply(rmsle_style, subset=["RMSLE"]).format(
        {"RMSLE": "{:.4f}", "R²": "{:.3f}", "MAE (s)": "{:.0f}", "Fit time (s)": "{:.1f}", "MAE (min)": "{:.1f}"})
    st.dataframe(styled, hide_index=True, width="stretch")
    n_train = bundle.metadata["rows"]["train"]
    n_val = bundle.metadata["rows"]["validation"]
    st.caption(f"All models: same {n_train:,} training trips, same {n_val:,} validation trips. RMSLE colour "
               "bands (strong < 0.35 ≤ moderate < 0.45 ≤ weak) are a reading aid for this dataset, not a "
               "standard. The reference baseline predicts the training median for every trip. Validation "
               "guides model choice, so these scores can be slightly optimistic.")


def feature_importance(bundle) -> None:
    imp = bundle.metadata["feature_importance"][:10][::-1]
    fig = go.Figure(go.Bar(
        x=[i["percent"] for i in imp], y=[i["feature"] for i in imp], orientation="h", marker_color=SERIES_1,
        marker_line_width=0, text=[f"{i['percent']:.1f}%" for i in imp], textposition="outside",
        customdata=[FEATURE_DESCRIPTIONS.get(i["feature"], "") for i in imp],
        hovertemplate="<b>%{y}</b><br>%{customdata}<br>Share of total gain: %{x:.2f}%<extra></extra>"))
    style_fig(fig, height=360, title="Top 10 features · XGBoost total gain", bargap=0.35)
    fig.update_xaxes(title="Share of total gain (%)", range=[0, max(i["percent"] for i in imp) * 1.2])
    st.plotly_chart(fig, width="stretch")
    st.caption("Gain = total loss reduction from splits on the feature. It shows what the model relied on, "
               "not causal effects; correlated features (e.g. the two distances) share credit unpredictably.")


def learning_curves(bundle) -> None:
    c = bundle.diagnostics["curves"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[p["iteration"] for p in c["train"]], y=[p["value"] for p in c["train"]],
                             name=f"Training (fixed {c['train_curve_rows']:,}-row subset)", mode="lines",
                             line=dict(color=SERIES_1, width=2)))
    fig.add_trace(go.Scatter(x=[p["iteration"] for p in c["validation"]], y=[p["value"] for p in c["validation"]],
                             name="Validation", mode="lines", line=dict(color=SERIES_2, width=2)))
    if c.get("best_iteration") is not None:
        fig.add_vline(x=c["best_iteration"] + 1, line_dash="dot", line_color=MUTED,
                      annotation_text=f"best iteration {c['best_iteration'] + 1}", annotation_font_color=MUTED)
    style_fig(fig, height=340, title="Learning curves · RMSLE per boosting round", hovermode="x unified")
    fig.update_xaxes(title="Boosting round")
    fig.update_yaxes(title="RMSLE")
    st.plotly_chart(fig, width="stretch")
    stopped_early = c.get("best_iteration") is not None and c["best_iteration"] + 1 < c["rounds_run"]
    if stopped_early:
        note = (f"Early stopping ended training after {c['rounds_run']} rounds; the model keeps the best validation "
                f"round ({c['best_iteration'] + 1}).")
    else:
        note = (f"All {c['rounds_run']} rounds ran without triggering early stopping (validation was still "
                "improving at the cap), so more rounds or a higher learning rate might help.")
    st.caption(note + " A widening gap between the lines would indicate overfitting.")


def distributions(bundle) -> None:
    d = bundle.diagnostics
    h = d["duration_histogram"]
    edges = h["edges"]
    mids = [(a + b) / 2 for a, b in zip(edges, edges[1:])]
    fig = go.Figure(go.Bar(x=mids, y=h["counts"], width=[(b - a) * 0.92 for a, b in zip(edges, edges[1:])],
                           marker_color=SERIES_1, marker_line_width=0,
                           customdata=[[a, b] for a, b in zip(edges, edges[1:])],
                           hovertemplate="%{customdata[0]:.1f}–%{customdata[1]:.1f} min<br>%{y:,} trips<extra></extra>"))
    style_fig(fig, height=300, title="Trip duration (training sample)")
    fig.update_xaxes(title="Minutes")
    fig.update_yaxes(title="Trips")
    hp = d["hourly_pickups"]
    fig2 = go.Figure()
    for lo, hi in ((7, 9), (16, 19)):
        fig2.add_vrect(x0=lo - 0.5, x1=hi + 0.5, fillcolor=WARNING, opacity=0.10, line_width=0)
    fig2.add_trace(go.Bar(x=list(range(24)), y=hp["weekday"], name="Weekday", marker_color=SERIES_1,
                          marker_line_width=0, hovertemplate="%{x}:00 weekday<br>%{y:,} pickups<extra></extra>"))
    fig2.add_trace(go.Bar(x=list(range(24)), y=hp["weekend"], name="Weekend", marker_color=SERIES_3,
                          marker_line_width=0, hovertemplate="%{x}:00 weekend<br>%{y:,} pickups<extra></extra>"))
    style_fig(fig2, height=300, title="Pickups by hour (shaded: weekday rush hours 07–09, 16–19)",
              barmode="stack", bargap=0.15)
    fig2.update_xaxes(title="Hour of day", dtick=2)
    fig2.update_yaxes(title="Pickups")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Long right tail: {h['above_range']:,} trips exceed 60 min and {h['below_range']:,} are under "
                   "1 min (outside the chart). This skew is why the model learns log duration.")
    with c2:
        st.plotly_chart(fig2, width="stretch")
        st.caption("The rush-hour flag only fires on weekdays, so weekend trips in shaded hours are not flagged.")


def render(bundle) -> None:
    kpi_tiles(bundle)
    benchmark_matrix(bundle)
    c1, c2 = st.columns(2)
    with c1:
        feature_importance(bundle)
    with c2:
        learning_curves(bundle)
    st.subheader("Dataset explorer")
    distributions(bundle)
