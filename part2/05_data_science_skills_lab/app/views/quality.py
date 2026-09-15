"""F11/S07 - data-quality audit workspace."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import glossary_expander, metric_tile, workspace_error
from skills_lab.benchmarks.quality import DIMENSION_DEFINITIONS, HEALTH_RULE
from skills_lab.catalog import View


def render(lab) -> None:
    result = lab.result(View.QUALITY)
    if result is None:
        workspace_error(lab.error(View.QUALITY) or "Result unavailable.")
        return

    st.subheader("Automated data-quality audit")
    st.caption(
        f"{result.dataset}. Every dimension below is a rate, so the score means the same "
        "thing on 500 rows as on a million."
    )

    tiles = st.columns(4)
    with tiles[0]:
        metric_tile("Composite score", f"{result.score:.1f} / 100")
    with tiles[1]:
        metric_tile("Completeness", f"{result.completeness:.2f}%")
    with tiles[2]:
        # Two decimals: a dimension with real violations must not round to a clean 100%.
        metric_tile("Validity", f"{result.validity:.2f}%")
    with tiles[3]:
        metric_tile("Duplicate rows", f"{result.duplicate_rows:,}")

    badge = {"PASS": "green", "WARNING": "orange", "CRITICAL": "red"}[result.health]
    st.markdown(f"**Overall state:** :{badge}[{result.health}]")

    failing = [c for c in result.column_reports if c.health != "PASS"]
    if failing:
        st.markdown(
            "**Columns needing work:** "
            + ", ".join(f"{c.column} ({c.health}, score {c.score:.1f})" for c in failing)
        )
        st.caption(
            "These are real defects, not review flags. A healthy dataset-level score can "
            "still hide a column that is unusable on its own."
        )

    with st.expander("How the score is built"):
        weights = ", ".join(f"{k} {v:.2f}" for k, v in result.weights.items())
        st.markdown(f"Weights: {weights}. " + HEALTH_RULE)
        for name, definition in DIMENSION_DEFINITIONS.items():
            st.markdown(f"- **{name.title()}** - {definition}")

    st.markdown("#### Per-column integrity")
    frame = pd.DataFrame(
        [
            {
                "Column": r.column,
                "Type": r.dtype,
                "Distinct": r.distinct,
                "Nulls": r.nulls,
                "Null %": round(r.null_pct, 2),
                "Invalid": r.invalid,
                "Inconsistent": r.inconsistent,
                "Score": round(r.score, 1),
                "Health": r.health,
                "Defects found": "; ".join(r.issues) or "no defects found",
            }
            for r in result.column_reports
        ]
    )

    def colour(value: str) -> str:
        return {
            "PASS": "background-color: rgba(0,190,90,0.30)",
            "WARNING": "background-color: rgba(240,170,0,0.35)",
            "CRITICAL": "background-color: rgba(230,60,60,0.35)",
        }.get(value, "")

    st.dataframe(
        frame.style.map(colour, subset=["Health"]),
        width="stretch",
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.markdown("#### Review flags - not defects")
        st.caption(
            "Things a human should look at. Statistical outliers and negative quantities "
            "are not automatically wrong."
        )
        flags = [f for f in result.review_flags if not f.startswith("Columns in CRITICAL")]
        if flags:
            for flag in flags:
                st.markdown(f"- {flag}")
        else:
            st.write("None.")
    with right:
        st.markdown("#### Business events in this slice")
        events = result.business_events
        st.markdown(
            f"- Credit notes: **{events['credit_notes']:,}**\n"
            f"- Return lines: **{events['return_lines']:,}**\n"
            f"- Non-product adjustment lines: **{events['adjustment_lines']:,}**\n"
            f"- Value of returns: **{events['returned_value']:,.2f}**"
        )
        st.caption(events["note"])

    glossary_expander(["Completeness", "Validity", "Uniqueness", "Consistency"])
