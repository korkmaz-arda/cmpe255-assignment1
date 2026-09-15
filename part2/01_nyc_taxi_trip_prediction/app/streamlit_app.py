"""NYC Taxi Trip Duration — ride estimator and data-science console.

Run: streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="NYC Taxi Trip Duration", page_icon="🚕", layout="wide")

from app import admin_autoresearch, admin_benchmarks, admin_deepdive, estimator, state  # noqa: E402
from app.dialogs import report_dialog, retrain_dialog  # noqa: E402
from app.ui_common import glossary_expander, inject_css  # noqa: E402


def header(bundle, error) -> None:
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        st.title("🚕 NYC Taxi Trip Duration")
        if bundle:
            vm = bundle.metadata["validation_metrics"]
            final = ("<span class='pill pill-good'>Finalized · test RMSLE "
                     f"{bundle.holdout['metrics']['rmsle']:.4f}</span>") if bundle.holdout else \
                "<span class='pill pill-warn'>Not finalized</span>"
            st.markdown(f"<span class='pill'>● {bundle.metadata['active_model']} · <span class='mono'>{bundle.version}"
                        f"</span></span><span class='pill'>Validation R² <span class='mono'>{vm['r2']:.3f}</span>"
                        f"</span>{final}", unsafe_allow_html=True)
        elif error:
            st.error(error)
        else:
            st.markdown("<span class='pill pill-warn'>● No model loaded</span>", unsafe_allow_html=True)
    with right:
        b1, b2 = st.columns(2)
        if b1.button("📑 CRISP-DM report", width="stretch"):
            report_dialog(bundle, state.latest_session())
        if b2.button("🛠️ Retrain", width="stretch", type="primary"):
            retrain_dialog()


def main() -> None:
    inject_css()
    bundle, error = state.active_bundle()
    header(bundle, error)
    if st.session_state.pop("celebrate", None):
        st.balloons()
        st.toast("New model version promoted. All views now reflect it.", icon="✅")
    if not state.trip_store().exists():
        st.warning("Processed data not found. Place the Kaggle archive at `data/raw/nyc-taxi-trip-duration.zip` "
                   "and run `python scripts/prepare_data.py` (see README). Retraining and AutoResearch need it.")

    tab_est, tab_admin = st.tabs(["🚖 Ride estimator", "🧪 Data-science console"])
    with tab_est:
        estimator.render(bundle)
    with tab_admin:
        sub_bench, sub_ar, sub_deep = st.tabs(["Benchmarks & training", "AutoResearch", "Deep-dive"])
        with sub_bench:
            if bundle:
                admin_benchmarks.render(bundle)
            else:
                st.info("Train a model to see benchmarks.")
        with sub_ar:
            admin_autoresearch.render()
        with sub_deep:
            if bundle:
                admin_deepdive.render(bundle)
            else:
                st.info("Train a model to see diagnostics.")
        glossary_expander()


main()
