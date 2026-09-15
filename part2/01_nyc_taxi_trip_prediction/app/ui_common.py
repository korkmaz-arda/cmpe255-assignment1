"""Shared UI helpers: styling, chart theme, formatting, glossary."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# Reference dark palette slots (validated categorical order) + status colours.
SERIES_1 = "#3987e5"   # blue
SERIES_2 = "#d95926"   # orange
SERIES_3 = "#199e70"   # aqua
GOOD, WARNING, CRITICAL = "#0ca30c", "#fab219", "#d03b3b"
INK_2, MUTED, GRID, AXIS = "#c3c2b7", "#898781", "#2c2c2a", "#383835"
TAXI_YELLOW = "#f7c600"

GLOSSARY = {
    "RMSLE": "Root mean squared logarithmic error: the typical error in log(1 + seconds). It measures relative "
             "error, so 0.30 means predictions are typically off by roughly ±35% (e^0.3 ≈ 1.35). Lower is better.",
    "R²": "Coefficient of determination in seconds: the share of trip-time variance the model explains. 1 is "
          "perfect; 0 is no better than predicting the average; below 0 is worse than that.",
    "MAE": "Mean absolute error: the average number of seconds a prediction is off, ignoring direction.",
    "MAPE": "Mean absolute percentage error: the average error as a percentage of the true duration.",
    "Validation split": "A fixed 15% of trips used to choose between models and settings. Because decisions "
                        "are made on it, its scores can be slightly optimistic.",
    "Calibration split": "A fixed 5% of trips used only to set the width of the prediction band.",
    "Final holdout (test)": "A fixed 15% of trips scored once, after the configuration is settled, for an "
                            "unbiased estimate. Retraining and AutoResearch never use it.",
    "Prediction band": "Empirical 90% range: on calibration trips of a similar distance tier, 90% of actual "
                       "durations fell within these offsets from the prediction. It is not a confidence interval "
                       "for the mean.",
    "Early stopping": "Boosting stops adding trees once validation error has not improved for 30 rounds.",
    "Temporal extrapolation": "The requested ride time lies outside the Jan–Jun 2016 period the model learned "
                              "from, so traffic patterns may differ.",
    "Hill climbing": "A greedy search: try one change, keep it only if validation RMSLE improves by more than "
                     "ε = 0.0001, otherwise revert it.",
    "City-block distance": "East–west plus north–south distance, approximating travel along a street grid.",
    "Great-circle distance": "Straight-line distance over the Earth's surface (haversine formula).",
}

CSS = """
<style>
div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
.hero { font-size: 2.6rem; font-weight: 700; line-height: 1.1; font-variant-numeric: tabular-nums; }
.hero-sub { color: #c3c2b7; font-size: 0.9rem; }
.pill { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem;
        border: 1px solid rgba(255,255,255,0.18); margin-right: 6px; }
.pill-good { border-color: #0ca30c; color: #7fd67f; }
.pill-warn { border-color: #fab219; color: #fab219; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }
.fare-row { display: flex; justify-content: space-between; padding: 3px 0;
            border-bottom: 1px solid rgba(255,255,255,0.06); }
.fare-total { display: flex; justify-content: space-between; padding: 6px 0; font-weight: 700; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS + TIME_CSS, unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 320, **layout) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=72, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK_2, size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        title_y=0.97, title_yanchor="top",
        hoverlabel=dict(bgcolor="#1a1a19", font_color="#ffffff"),
        **layout,
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def fmt_seconds(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}m {s % 60:02d}s"


def glossary_expander() -> None:
    with st.expander("Glossary: what do these terms mean?"):
        for term, text in GLOSSARY.items():
            st.markdown(f"**{term}** — {text}")


TIME_CSS = """
<style>
/* Streamlit renders time/date as segmented react-aria fields (no <input>): enlarge the segments. */
.st-key-pickup_time [data-testid="stTimeInputTimeDisplay"] [role="spinbutton"],
.st-key-pickup_time [data-testid="stTimeInputTimeDisplay"] [data-type="literal"] {
  font-size: 2.1rem; font-weight: 700; color: #f7c600; font-variant-numeric: tabular-nums; line-height: 1.2; }
.st-key-pickup_time [data-testid="stTimeInputTimeDisplay"] [role="spinbutton"]:focus {
  background: rgba(247,198,0,0.18); border-radius: 4px; outline: none; }
.st-key-ride_date [role="spinbutton"], .st-key-ride_date [data-type="literal"] {
  font-size: 1.25rem; font-variant-numeric: tabular-nums; line-height: 1.6; }
.st-key-pickup_time label p, .st-key-ride_date label p { font-size: 1.05rem; font-weight: 700; }
.st-key-time_shift button p { font-size: 0.8rem; white-space: nowrap; }
</style>
"""
