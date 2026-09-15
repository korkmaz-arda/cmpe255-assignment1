"""S03/S04 - the two dialogs: skill execution output and the methodology report."""

from __future__ import annotations

import json

import numpy as np
import streamlit as st

from app.report import build_report


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value


@st.dialog("Skill execution result", width="large")
def skill_result_dialog(payload: dict) -> None:
    st.success(
        f"**{payload['skill']}** ran against: {payload['computation']}"
    )
    st.caption(
        "This is the complete result payload, exactly as the analytical package returned it."
    )
    st.code(json.dumps(_json_safe(payload), indent=2), language="json")
    if st.button("Close", width="stretch"):
        st.session_state.pop("pending_skill", None)
        st.rerun()


@st.dialog("CRISP-DM methodology report", width="large")
def methodology_dialog(lab) -> None:
    st.caption(
        "Every figure below is read from the results currently in memory - this report "
        "cannot disagree with the numbers on screen."
    )
    for heading, body in build_report(lab):
        st.subheader(heading)
        st.markdown(body)
    if st.button("Close", width="stretch"):
        st.session_state.pop("show_report", None)
        st.rerun()
