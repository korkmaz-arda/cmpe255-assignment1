"""Application shell: header, view switching, modals and retraining studio (S13, F11)."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Customer Segmentation Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from segmentation import config  # noqa: E402

from app import admin, crispdm, explorer, state, theme  # noqa: E402


def _header(engine) -> None:
    meta = engine.meta
    online = engine.trained
    dataset = meta.get("dataset", {}).get("name", config.DATASET_NAME)
    detail = (
        f"{meta.get('n_customers', 0):,} customers · k={meta.get('k', '—')} · seed {meta.get('seed', '—')}"
        if online else "no artifacts loaded"
    )
    st.markdown(
        theme.compact(
        f"""<div class='app-header'>
          <div>
            <h1>🎯 Customer Segmentation Intelligence</h1>
            <div class='sub'>Unsupervised retail segmentation · {dataset}</div>
          </div>
          <div style='text-align:right;font-size:.82rem;'>
            <div><span class='status-dot {"" if online else "off"}'></span>
              {"Model online" if online else "No model"}</div>
            <div style='opacity:.8'>{detail}</div>
          </div>
        </div>"""
        ),
        unsafe_allow_html=True,
    )


@st.dialog("Retraining studio", width="large")
def _retrain_studio(engine) -> None:
    st.caption(
        "Re-runs the entire pipeline — preparation, tournament, production fit, projections, "
        "persona profiling, elbow sweep and artifact export — then hot-reloads the serving layer. "
        "Every control below actually reaches the pipeline."
    )
    available = int(engine.meta.get("n_available", 2197))

    k = st.slider("Cluster count (k)", config.MIN_K, config.MAX_K,
                  int(engine.meta.get("k", config.DEFAULT_K)))
    rows = st.slider(
        "Training sample (customers)", config.MIN_TRAIN_SAMPLE, available, available, step=100,
        help="The real dataset is a fixed size, so the meaningful analogue of a dataset-size "
             "control is how much of it the model trains on. Drawn deterministically from the seed.",
    )
    seed = st.number_input("Random seed", 0, 10_000, int(engine.meta.get("seed", config.DEFAULT_SEED)))

    st.markdown(
        f"<div class='note'>Persona identities are matched to cluster <i>content</i>, so changing k "
        f"re-derives which personas are in play rather than shuffling fixed labels between "
        f"clusters. At higher k some segments may fit no persona; those are described from their "
        f"own data instead. Up to {config.MAX_K} clusters are supported.</div>",
        unsafe_allow_html=True,
    )

    if st.button("Retrain now", type="primary", width="stretch"):
        try:
            with st.spinner("Re-running the full pipeline…"):
                state.retrain(k=int(k), seed=int(seed),
                              n_rows=None if rows >= available else int(rows))
            st.session_state.prediction = None
            st.session_state.selected_cluster = None
            st.session_state.action_message = (
                "success", f"Retrained on {rows:,} customers with k={k}. All views refreshed."
            )
            st.balloons()
            st.rerun()
        except Exception as exc:
            st.error(f"Retraining failed: {exc}")


@st.dialog("CRISP-DM methodology report", width="large")
def _crispdm_modal(engine) -> None:
    crispdm.render(engine)


def main() -> None:
    theme.inject()
    state.init_session()
    engine = state.get_engine()

    _header(engine)

    message = st.session_state.action_message
    if message:
        getattr(st, message[0])(message[1])
        st.session_state.action_message = None

    controls = st.columns([2, 2, 1.3, 1.3, 2.4])
    with controls[0]:
        st.segmented_control(
            "View", state.VIEWS, key="view_control", label_visibility="collapsed",
        )
    if controls[2].button("📋 Methodology", width="stretch"):
        _crispdm_modal(engine)
    if controls[3].button("⚙ Retrain", width="stretch"):
        _retrain_studio(engine)

    st.divider()

    if state.current_view() == state.VIEW_EXPLORER:
        explorer.render(engine)
    else:
        admin.render(engine)


main()
