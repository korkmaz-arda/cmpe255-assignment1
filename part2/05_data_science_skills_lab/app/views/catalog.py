"""F02/F12/S02/S03 - the skills catalog: browse, search, filter, run, jump."""

from __future__ import annotations

import streamlit as st

from app.components import glossary_expander
from skills_lab.catalog import CATEGORIES, search, skill_count


ALL = "All categories"


def render(lab) -> None:
    st.subheader(f"Skills catalog ({skill_count()} skills)")
    st.caption(
        "Each skill explains what a technique is for, the statistical idea behind it, and "
        "the mistakes it usually invites. Every skill also runs: 'Run this skill' executes "
        "the analysis it describes and shows you the real result payload."
    )

    left, right = st.columns([3, 2])
    with left:
        query = st.text_input(
            "Search", placeholder="Search name, purpose or source...", key="catalog_query"
        )
    with right:
        category = st.selectbox("Category", [ALL, *CATEGORIES], key="catalog_category")

    results = search(query, None if category == ALL else category)
    st.caption(f"{len(results)} of {skill_count()} skills match.")

    if not results:
        st.info("No skill matches that search. Clear the box or pick another category.")
        return

    columns = st.columns(2)
    for index, skill in enumerate(results):
        with columns[index % 2]:
            _card(skill, lab)

    glossary_expander()


def _card(skill, lab) -> None:
    with st.container(border=True):
        st.caption(f"{skill.category}  ·  source: {skill.origin}")
        st.markdown(f"### {skill.name}")
        st.write(skill.purpose)

        st.markdown("**Statistical idea**")
        st.info(skill.intuition)

        st.markdown("**Common pitfalls**")
        for pitfall in skill.pitfalls:
            st.markdown(f"- :orange[{pitfall}]")

        st.caption(f"Demonstrated in: {skill.workspace_label}")
        run_col, jump_col = st.columns(2)
        with run_col:
            if st.button(
                "Run this skill", key=f"run-{skill.id}", width="stretch"
            ):
                st.session_state["pending_skill"] = skill.id
                st.rerun()
        with jump_col:
            if st.button(
                "Open workspace", key=f"jump-{skill.id}", width="stretch"
            ):
                st.session_state["view"] = skill.view.value
                st.rerun()
