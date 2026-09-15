"""Admin: deep-dive diagnostics computed on the validation split for the active version (S04)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from taxi.features import FEATURE_DESCRIPTIONS

from app.ui_common import GLOSSARY, SERIES_1, SERIES_2, MUTED, style_fig


def render(bundle) -> None:
    dd = bundle.diagnostics["deep_dive"]
    st.markdown(f"All values below are **computed on the fixed validation split** ({dd['n']:,} trips) for model "
                f"version `{bundle.version}`. They are recalculated at every retrain.")

    st.subheader("Accuracy by distance tier (validation)")
    tiers = pd.DataFrame([{"Tier": t["label"], "MAE (s)": t["mae_s"], "MAPE (%)": t["mape_pct"], "R²": t["r2"],
                           "RMSLE": t["rmsle"], "Validation trips": t["n"]} for t in dd["tiers"]])
    st.dataframe(tiers.style.format({"MAE (s)": "{:.0f}", "MAPE (%)": "{:.1f}", "R²": "{:.3f}", "RMSLE": "{:.4f}",
                                     "Validation trips": "{:,}"}), hide_index=True, width="stretch")
    st.caption("Tiers use great-circle distance; a trip with either end within 2 km of JFK, LaGuardia or Newark "
               "counts as Airport regardless of distance. R² within a tier is measured against that tier's own, "
               "narrower spread of durations, so it is not directly comparable to the overall R². "
               + GLOSSARY["MAPE"])

    st.subheader("Residual diagnostics (validation)")
    r = dd["residuals"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skewness (log residual)", f"{r['skewness']:+.3f}", "ideal 0", delta_color="off", delta_arrow="off",
              help="Asymmetry of errors. Positive = a longer tail of over-predictions.")
    c2.metric("Excess kurtosis (log residual)", f"{r['excess_kurtosis']:+.2f}", "ideal 0 (normal)",
              delta_color="off", delta_arrow="off", help="Heavier-than-normal tails mean more extreme misses.")
    c3.metric("Mean bias error", f"{r['mean_bias_error_s']:+.0f} s", "ideal 0", delta_color="off", delta_arrow="off",
              help="Average predicted minus actual seconds. Negative = under-predicts on average.")
    c4.metric("MAPE", f"{r['mape_pct']:.1f}%", "ideal 0%", delta_color="off", delta_arrow="off", help=GLOSSARY["MAPE"])
    if r["mean_bias_error_s"] < 0:
        st.caption("The negative mean bias fits a model trained in log space: its back-transformed predictions "
                   "sit nearer the median than the mean of right-skewed durations.")

    cov = bundle.metadata.get("band_validation_coverage")
    if cov:
        st.caption(f"Prediction-band coverage on validation (descriptive): {cov['coverage']:.1%} of trips fell "
                   f"inside the band (nominal {cov['nominal']:.0%}); the band was fitted on the separate "
                   "calibration split.")

    d = bundle.diagnostics
    sample = pd.DataFrame(d["residual_sample"]["rows"])
    c1, c2 = st.columns(2)
    with c1:
        h = d["residual_histogram"]
        e = h["edges"]
        fig = go.Figure(go.Bar(x=[(a + b) / 2 for a, b in zip(e, e[1:])], y=h["counts"], marker_color=SERIES_1,
                               width=[(b - a) * 0.92 for a, b in zip(e, e[1:])], marker_line_width=0,
                               customdata=[[a, b] for a, b in zip(e, e[1:])],
                               hovertemplate="%{customdata[0]:+.0f} to %{customdata[1]:+.0f} min<br>%{y:,} trips<extra></extra>"))
        style_fig(fig, height=300, title="Residuals: predicted − actual (validation)")
        fig.update_xaxes(title="Minutes")
        fig.update_yaxes(title="Trips")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Outside the ±15 min window: {h['below_range']:,} below, {h['above_range']:,} above.")
    with c2:
        top = max(sample["actual_min"].max(), sample["predicted_min"].max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", name="Perfect prediction",
                                 line=dict(color=MUTED, dash="dot", width=1), hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sample["actual_min"], y=sample["predicted_min"], mode="markers",
                                 name="Validation trip", marker=dict(color=SERIES_2, size=8, opacity=0.6,
                                                                     line=dict(color="#1a1a19", width=1)),
                                 hovertemplate="actual %{x:.1f} min<br>predicted %{y:.1f} min<extra></extra>"))
        style_fig(fig, height=300, title=f"Predicted vs actual ({len(sample)} random validation trips)")
        fig.update_xaxes(title="Actual minutes")
        fig.update_yaxes(title="Predicted minutes")
        st.plotly_chart(fig, width="stretch")

    st.subheader("Feature correlation with log duration (validation)")
    def p_text(p):
        if p is None:
            return "—"
        return "< 1e-300 (underflow)" if p == 0 else f"{p:.1e}"

    corr = pd.DataFrame([{"Feature": c["feature"], "Meaning": FEATURE_DESCRIPTIONS.get(c["feature"], ""),
                          "Pearson r": c["pearson_r"], "Signal": c["signal"], "p-value": p_text(c["p_value"])}
                         for c in dd["correlations"]])
    st.dataframe(corr.style.format({"Pearson r": "{:+.3f}"}, na_rep="—"),
                 hide_index=True, width="stretch")
    st.caption("Pearson r measures *linear* association only; tree models also use non-linear patterns (e.g. "
               "coordinates). With ~200k trips almost every p-value is tiny, so read the size of r, not the "
               "p-value. Signal labels: |r| ≥ 0.5 strong, ≥ 0.3 moderate, ≥ 0.1 weak, else negligible.")
