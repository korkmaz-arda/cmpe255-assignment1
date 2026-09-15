"""F08/F09/F10/S07 - cohort retention, checkout funnel, revenue series, A/B calculator."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components import glossary_expander, metric_tile, workspace_error
from skills_lab.catalog import View
from skills_lab.stats import ASSUMPTIONS, ABInputError, two_proportion_test


def render(lab) -> None:
    result = lab.result(View.ANALYTICS)
    if result is None:
        workspace_error(lab.error(View.ANALYTICS) or "Result unavailable.")
        return

    st.subheader("Business and statistical analytics")
    st.caption(
        "Cohort retention and the revenue series are computed from the real Online Retail II "
        "transaction log. The A/B test runs on counts you enter. The checkout funnel is the "
        "one illustrative element here - it is not derived from the retail data."
    )

    _retention(result)
    st.divider()
    left, right = st.columns([1, 1])
    with left:
        _funnel(result)
    with right:
        _revenue(result)
    st.divider()
    _ab_test()
    glossary_expander(
        ["Cohort", "Retention", "z-score", "p-value", "Confidence interval",
         "Percentage point (pp)"]
    )


def _retention(result) -> None:
    st.markdown("#### Cohort retention")
    st.caption(result.retention.population_rule)

    matrix = result.retention.matrix.copy()
    matrix.columns = [f"m{int(c)}" for c in matrix.columns]
    display = matrix.copy()
    display.insert(0, "Customers", result.retention.sizes.values)
    display.index.name = "Cohort"

    st.dataframe(
        display.style.background_gradient(
            cmap="Greens", subset=[c for c in matrix.columns], vmin=0, vmax=60
        ).format("{:.1f}", subset=[c for c in matrix.columns], na_rep=""),
        width="stretch",
    )
    st.caption(
        f"Every observable period is shown - {len(matrix.columns)} period columns, m0 to "
        f"m{len(matrix.columns) - 1}, which is as far as the oldest cohort can be followed. "
        "m0 is the acquisition month (always 100%). A cohort is blank in periods that had "
        "not happened yet when the data ends: blank means 'not observable', not 'zero'."
    )


def _funnel(result) -> None:
    st.markdown("#### Checkout funnel")
    st.warning(result.funnel.label, icon="⚠️")
    frame = pd.DataFrame(result.funnel.stages)
    figure = go.Figure(
        go.Funnel(
            y=frame["stage"],
            x=frame["users"],
            textinfo="value+percent initial",
        )
    )
    figure.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figure, width="stretch")
    worst = frame.iloc[1:]["step_dropoff_pct"].idxmax()
    st.caption(
        f"Largest single drop-off: {frame.loc[worst, 'stage']} "
        f"({frame.loc[worst, 'step_dropoff_pct']:.1f}% of the previous stage lost). "
        "These counts are a teaching example, not measured telemetry."
    )


def frame_span(result) -> str:
    dates = result.revenue.frame["date"]
    return f"{dates.min().date()} to {dates.max().date()}"


def _revenue(result) -> None:
    st.markdown("#### Daily revenue and its weekly pattern")
    st.caption(
        f"The last {result.revenue.days} days on which transactions were recorded "
        f"({frame_span(result)}). Calendar days with no transactions inside that span are "
        "treated as zero revenue by the decomposition."
    )
    frame = result.revenue.frame
    figure = px.line(
        frame,
        x="date",
        y=["net_revenue", "gross_sales", "trend"],
        labels={"value": "revenue", "date": "", "variable": ""},
    )
    figure.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"{result.revenue.definitions['net']} {result.revenue.definitions['gross']} "
        f"The trend line is the centred moving average from an additive decomposition with a "
        f"{result.revenue.seasonal_period}-day period; it is descriptive, not a forecast."
    )


def _ab_test() -> None:
    st.markdown("#### A/B significance calculator")
    st.caption(
        "Enter the counts from your own experiment. This project ships no stored experiment "
        "record and no example numbers - the fields start empty, and nothing is computed "
        "until you provide real counts."
    )

    control, treatment = st.columns(2)
    with control:
        st.markdown("**Control**")
        cv = st.number_input("Control visitors", 0, 10_000_000, 0, step=500, key="ab_cv")
        cc = st.number_input("Control conversions", 0, 10_000_000, 0, step=10, key="ab_cc")
        st.caption(f"Observed rate: {cc / cv * 100:.3f}%" if cv else "Awaiting your counts.")
    with treatment:
        st.markdown("**Treatment**")
        tv = st.number_input("Treatment visitors", 0, 10_000_000, 0, step=500, key="ab_tv")
        tc = st.number_input("Treatment conversions", 0, 10_000_000, 0, step=10, key="ab_tc")
        st.caption(f"Observed rate: {tc / tv * 100:.3f}%" if tv else "Awaiting your counts.")

    mde = st.slider(
        "Smallest difference worth shipping (percentage points)",
        0.0, 5.0, 0.5, 0.1,
        help="Your practical threshold. It shapes the recommendation and never changes the statistics.",
        key="ab_mde",
    )

    if not st.button("Run the test", type="primary"):
        st.info("Set the counts above, then run the test.")
        return

    if cv <= 0 or tv <= 0:
        st.error(
            "Enter your own visitor and conversion counts above first - both arms need a "
            "visitor count greater than zero. This project ships no example experiment."
        )
        return

    try:
        result = two_proportion_test(
            int(cv), int(cc), int(tv), int(tc), minimum_practical_effect_pp=mde
        )
    except ABInputError as exc:
        st.error(str(exc))
        return

    # Remember the counts the user actually submitted. Streamlit discards widget state for
    # widgets that are not rendered, so without this the A/B skill card could never see
    # them. These are the user's own numbers - nothing is invented here.
    st.session_state["ab_last_counts"] = {
        "control_visitors": int(cv),
        "control_conversions": int(cc),
        "treatment_visitors": int(tv),
        "treatment_conversions": int(tc),
    }

    tiles = st.columns(4)
    with tiles[0]:
        metric_tile("z-score", f"{result.z_score:.3f}")
    with tiles[1]:
        metric_tile("p-value", f"{result.p_value:.4g}")
    with tiles[2]:
        metric_tile(
            "Absolute lift",
            f"{result.absolute_lift_pp:+.2f} pp",
            help_key="Percentage point (pp)",
        )
    with tiles[3]:
        metric_tile("Relative lift", f"{result.relative_lift_pct:+.1f}%")

    st.markdown(
        f"**95% confidence interval on the difference:** "
        f"{result.ci_low_pp:+.2f} pp to {result.ci_high_pp:+.2f} pp"
    )
    if result.significant:
        st.success(f"**{result.recommendation}** — {result.rationale}")
    else:
        st.warning(f"**{result.recommendation}** — {result.rationale}")

    for warning in result.warnings:
        st.error(warning)

    with st.expander("Context and assumptions"):
        st.markdown(
            f"- Observed (post-hoc) power: **{result.observed_power:.2f}**. "
            f"{result.power_note}\n"
            f"- Significance level: α = {result.alpha}\n"
            f"- Pooled conversion rate: {result.pooled_rate * 100:.3f}%\n"
            f"- Pooled standard error: {result.standard_error:.6f}"
        )
        st.markdown("**This test assumes:**")
        for assumption in ASSUMPTIONS:
            st.markdown(f"- {assumption}")
