"""Market Basket Intelligence — dashboard entry point.

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import admin, crispdm, glossary, state, workspace  # noqa: E402
from basket import config  # noqa: E402

VIEWS = ("Basket workspace", "Data-science admin")
VIEW_ICONS = {"Basket workspace": "🛒", "Data-science admin": "🔬"}

st.set_page_config(
    page_title="Market Basket Intelligence",
    page_icon="🛒",
    layout="wide",
)


def main() -> None:
    state.init_session()

    if not state.artifacts_ready():
        _first_run()
        return

    engine = state.get_engine()
    _header(engine)

    # One widget, one key: the control's own state is the single source of truth for
    # which view is showing. Mirroring it into a second key would let the two drift.
    view = st.segmented_control(
        "View", VIEWS,
        key="view",
        required=True,
        format_func=lambda option: f"{VIEW_ICONS[option]}  {option}",
        label_visibility="collapsed",
    ) or VIEWS[0]
    st.divider()

    if view == VIEWS[0]:
        workspace.render(engine)
    else:
        admin.render(engine)

    _glossary_sidebar()


def _header(engine) -> None:
    health = engine.health()
    left, right = st.columns([5, 2])
    with left:
        st.title("🛒 Market Basket Intelligence")
        st.caption(
            f"Association rules mined from {health['n_transactions']:,} real "
            f"{config.DATASET_NAME} orders across {health['catalog_size']} products."
        )
    with right:
        st.success(
            f"**Service ready** · {health['n_rules']:,} rules loaded · "
            f"mined by {health['champion']}"
        )
        buttons = st.columns(2)
        if buttons[0].button("Methodology", width="stretch",
                             help="CRISP-DM report for this run"):
            crispdm.show(engine)
        if buttons[1].button("Refresh", width="stretch",
                             help="Re-read the artifacts from disk"):
            state.invalidate()
            st.rerun()


def _first_run() -> None:
    st.title("🛒 Market Basket Intelligence")
    st.warning("No mined artifacts yet — there is nothing to serve.")
    st.markdown(
        """
Run these once from the project directory:

```bash
python -m basket.data --download     # fetch Instacart, build catalog + corpus
python -m basket.pipeline            # mine the rule set and write the artifacts
python -m basket.autoresearch        # optional: run the parameter search
```
"""
    )
    if st.button("Check again"):
        state.invalidate()
        st.rerun()


def _glossary_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Glossary")
        st.caption("Every term this dashboard uses, in plain language.")
        for term, definition in glossary.TERMS.items():
            with st.expander(term):
                st.write(definition)


main()
