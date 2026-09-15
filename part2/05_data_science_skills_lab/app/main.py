"""S01/S05/S06/S08 - the six-view shell.

Run with:  streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from app import modals, state  # noqa: E402
from app.views import analytics, catalog, classification, fraud, quality, regression  # noqa: E402
from skills_lab.catalog import View, skill_count  # noqa: E402
from skills_lab.execute import NeedsUserCountsError, execute  # noqa: E402

VIEW_LABELS = {
    View.CATALOG: f"Skills catalog ({skill_count()})",
    View.CLASSIFICATION: "Classification",
    View.REGRESSION: "Regression",
    View.FRAUD: "Fraud (imbalanced)",
    View.ANALYTICS: "Business analytics",
    View.QUALITY: "Data quality",
}

RENDERERS = {
    View.CATALOG: catalog.render,
    View.CLASSIFICATION: classification.render,
    View.REGRESSION: regression.render,
    View.FRAUD: fraud.render,
    View.ANALYTICS: analytics.render,
    View.QUALITY: quality.render,
}


def header() -> None:
    left, right = st.columns([4, 1])
    with left:
        st.title("Data Science Skills Mastery Lab")
        st.caption(
            "A teaching workbench: a catalog of analytical techniques, each demonstrated on "
            "real public data by a workspace that actually computes its result."
        )
    with right:
        st.metric("Skills in catalog", skill_count())
        if st.button("Methodology report", width="stretch"):
            st.session_state["show_report"] = True
            st.rerun()


def data_gate(lab) -> bool:
    """Honest empty state: say what is missing and how to fix it (S06)."""
    if not lab.missing_data:
        return True
    st.warning(
        "The local data cache is incomplete, so the workspaces below cannot be computed. "
        f"Missing: {', '.join(lab.missing_data)}."
    )
    st.code("python scripts/prepare_data.py", language="bash")
    st.caption(
        "No stand-in figures are shown anywhere in this app: if a number is on screen, it "
        "was computed from the data on this machine."
    )
    return False


def main() -> None:
    st.set_page_config(
        page_title="Data Science Skills Mastery Lab",
        page_icon="🧪",
        layout="wide",
    )
    st.session_state.setdefault("view", View.CATALOG.value)

    header()

    with st.spinner("Computing all five benchmarks (first load only)..."):
        lab = state.load_lab()

    active = View(st.session_state["view"])
    _navigation(active)

    if lab.errors:
        with st.expander(f"{len(lab.errors)} workspace(s) failed to compute", expanded=False):
            for key, message in lab.errors.items():
                st.error(f"**{key}**: {message}")

    data_ready = data_gate(lab)
    if data_ready or active is View.CATALOG:
        RENDERERS[active](lab)

    _dialogs(lab)
    _sidebar(lab)


def _ab_inputs() -> dict[str, int] | None:
    """The user's A/B counts, or None if they have not entered any yet.

    Live widget values are preferred; the counts from their last run are the fallback,
    because Streamlit discards widget state while the analytics view is not on screen.
    """
    counts = {
        "control_visitors": int(st.session_state.get("ab_cv", 0) or 0),
        "control_conversions": int(st.session_state.get("ab_cc", 0) or 0),
        "treatment_visitors": int(st.session_state.get("ab_tv", 0) or 0),
        "treatment_conversions": int(st.session_state.get("ab_tc", 0) or 0),
    }
    if counts["control_visitors"] > 0 and counts["treatment_visitors"] > 0:
        return counts
    return st.session_state.get("ab_last_counts")


def _navigation(active: View) -> None:
    """Six-view navigation (S01). Plain buttons, so a jump from a catalog card can set the
    active view without colliding with a stateful navigation widget."""
    columns = st.columns(len(View))
    for column, view in zip(columns, View):
        with column:
            if st.button(
                VIEW_LABELS[view],
                key=f"nav-{view.value}",
                width="stretch",
                type="primary" if view is active else "secondary",
            ):
                st.session_state["view"] = view.value
                st.rerun()


def _dialogs(lab) -> None:
    if st.session_state.get("show_report"):
        modals.methodology_dialog(lab)
    skill_id = st.session_state.get("pending_skill")
    if skill_id:
        try:
            payload = execute(skill_id, lab.results, ab_inputs=_ab_inputs())
        except NeedsUserCountsError as exc:
            # The A/B skill has no dataset to run on: it needs counts from the user.
            st.session_state.pop("pending_skill", None)
            st.session_state["view"] = View.ANALYTICS.value
            st.info(str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - shown to the user verbatim
            st.session_state.pop("pending_skill", None)
            st.error(f"That skill could not run: {exc}")
            return
        st.balloons()
        modals.skill_result_dialog(payload)


def _sidebar(lab) -> None:
    with st.sidebar:
        st.markdown("### Service status")
        status = state.service_status()
        st.write(f"**{status['service']}**")
        st.write(f"Skills in catalog: {status['skills_in_catalog']}")
        for key, value in status["datasets"].items():
            icon = "✅" if value == "ready" else "❌"
            st.write(f"{icon} {key}: {value}")
        st.caption("Benchmarks computed this session: " f"{len(lab.results)} of 5")
        if st.button("Recompute everything"):
            state.reset()
            st.rerun()
        st.divider()
        st.caption(
            "Reimplementation of a data-science teaching lab specification, rebuilt on real "
            "public data (OpenML, UCI). No figure in this app is hardcoded."
        )


main()
