"""F05/S07 - house-price regression workspace."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components import glossary_expander, metric_tile, workspace_error
from skills_lab.catalog import View
from skills_lab.config import REGRESSION_TEST_SIZE, SEED
from skills_lab.data.ames import DISPLAY_NAMES


def render(lab) -> None:
    result = lab.result(View.REGRESSION)
    if result is None:
        workspace_error(lab.error(View.REGRESSION) or "Result unavailable.")
        return

    st.subheader("House-price regression - Ames, Iowa")
    st.caption(
        f"{result.dataset}. {result.rows:,} sales, hold-out "
        f"{int(REGRESSION_TEST_SIZE * 100)}% (seed {SEED}). The forest is fitted on "
        "log1p(price) because prices are right-skewed; predictions are back-transformed "
        "with expm1 before scoring, so every error below is in dollars."
    )

    tiles = st.columns(3)
    with tiles[0]:
        metric_tile("R-squared", f"{result.r2:.3f}", help_key="R-squared")
    with tiles[1]:
        metric_tile("RMSE", f"${result.rmse:,.0f}")
    with tiles[2]:
        metric_tile("MAE", f"${result.mae:,.0f}")
    st.caption(
        f"For scale, the median sale price is ${result.price_median:,.0f}, so the typical "
        f"miss is about {result.mae / result.price_median * 100:.0f}% of a median house."
    )

    left, right = st.columns([3, 2])
    with left:
        _calibration(result)
    with right:
        _importances(result)

    _samples(result)
    st.caption(
        "Living area and basement area correlate at "
        f"{result.area_basement_correlation:.2f} in this data, so their importances "
        "partly substitute for one another."
    )
    glossary_expander(["RMSE", "MAE", "R-squared", "Leakage"])


def _calibration(result) -> None:
    st.markdown("#### Actual versus predicted sale price")
    frame = pd.DataFrame(result.predictions)
    st.caption(
        f"The first {len(frame)} of {result.test_rows} held-out sales, in split order "
        "(not selected by price or error)."
    )
    low = float(min(frame["actual"].min(), frame["predicted"].min()))
    high = float(max(frame["actual"].max(), frame["predicted"].max()))
    figure = px.scatter(
        frame,
        x="actual",
        y="predicted",
        hover_data={"actual": ":$,.0f", "predicted": ":$,.0f", "residual": ":$,.0f"},
        labels={"actual": "actual price (USD)", "predicted": "predicted price (USD)"},
        opacity=0.7,
    )
    figure.add_trace(
        go.Scatter(
            x=[low, high], y=[low, high], mode="lines", name="perfect prediction",
            line=dict(dash="dash", color="grey"),
        )
    )
    figure.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Points on the dashed line are exact. Points below it are under-predictions - the "
        "forest is conservative about the most expensive houses."
    )


def _importances(result) -> None:
    st.markdown("#### Feature importance")
    frame = pd.DataFrame(
        [
            {"feature": DISPLAY_NAMES.get(i.feature, i.feature), "share": i.share_pct}
            for i in result.importances
        ]
    )
    figure = px.bar(
        frame.sort_values("share"),
        x="share",
        y="feature",
        orientation="h",
        labels={"share": "share of total importance (%)", "feature": ""},
    )
    figure.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figure, width="stretch")


def _samples(result) -> None:
    with st.expander("Sample held-out rows"):
        frame = pd.DataFrame(result.sample_rows).rename(columns=DISPLAY_NAMES)
        st.dataframe(frame.astype(str), width="stretch")
