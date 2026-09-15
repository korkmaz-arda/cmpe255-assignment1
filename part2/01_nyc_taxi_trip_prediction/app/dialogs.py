"""Modal dialogs: retraining studio (F13), CRISP-DM report (S05), step inspection (S07)."""
from __future__ import annotations

import json

import streamlit as st

from taxi import config
from taxi.report import PHASES, build_report
from taxi.train import TrainParams, run_training

from app import state


@st.dialog("Step detail", width="large")
def step_dialog(step: dict) -> None:
    st.markdown(f"### #{step['iteration']} · {step['component']}")
    st.caption(f"Phase: {step['phase']} · {step['category']} · {step['timestamp']}")
    c1, c2, c3 = st.columns(3)
    if step["decision"] == "BENCHMARK":
        c1.metric("Validation RMSLE", f"{step['rmsle_candidate']:.5f}")
        c2.metric("Decision", "Benchmark (not gated)")
    else:
        c1.metric("Score before (incumbent)", f"{step['incumbent_before']:.5f}")
        c2.metric("Candidate score", f"{step['rmsle_candidate']:.5f}", f"{-step['delta']:+.5f}", delta_color="inverse")
        c3.metric("Decision", ("✅ " if step["decision"] == "ACCEPTED" else "❌ ") + step["decision"],
                  f"incumbent after {step['incumbent_after']:.5f}", delta_color="off", delta_arrow="off")
    st.markdown(f"**Hypothesis.** {step['hypothesis']}")
    st.markdown("**Code change** (the exact expression the search executed)")
    st.code(step["code"], language="python")
    st.markdown("**Reflection** · *auto-generated from the measured scores*")
    st.write(step["reflection"])
    st.markdown("**Active hyperparameters**")
    st.code(json.dumps(step["params"], indent=2), language="json")
    if step.get("features"):
        with st.expander(f"Feature set tried ({len(step['features'])} columns)"):
            st.write(", ".join(step["features"]))


@st.dialog("Retraining studio", width="large")
def retrain_dialog() -> None:
    trips = state.trip_store()
    if not trips.exists():
        st.error("Processed data not found. Run `python scripts/prepare_data.py` first.")
        return
    report = trips.report() or {}
    pool = report.get("split_sizes", {}).get("fit")
    b = config.RETRAIN_BOUNDS
    st.markdown("Runs the **full pipeline**: sample the fit pool → engineer features → fit the median baseline, "
                "Ridge, random forest and XGBoost → calibrate the band → compute diagnostics → write a new "
                "version → check it loads and predicts → switch the live pointer. Scores are on the fixed "
                "validation split; the test split is not read. The new version starts *not finalized*.")
    with st.form("retrain_form"):
        max_size = pool or 1_000_000
        sample = st.slider("Training sample size (trips from the fit pool)", b["sample_size"][0], max_size,
                           min(200_000, max_size), step=10_000,
                           help="Seeded random sample. Validation and calibration rows never change.")
        c1, c2 = st.columns(2)
        rounds = c1.slider("Boosting rounds (maximum)", *b["n_estimators"], config.DEFAULT_XGB["n_estimators"], step=10,
                           help="Early stopping may use fewer.")
        depth = c2.slider("Tree depth", *b["max_depth"], config.DEFAULT_XGB["max_depth"])
        lr = c1.number_input("Learning rate", min_value=b["learning_rate"][0], max_value=b["learning_rate"][1],
                             value=config.DEFAULT_XGB["learning_rate"], step=0.01, format="%.2f")
        sub = c2.slider("Row subsample fraction", *b["subsample"], config.DEFAULT_XGB["subsample"], step=0.05)
        seed = c1.number_input("Seed", min_value=0, max_value=100_000, value=42, step=1)
        running = st.session_state.get("retrain_running", False)
        submitted = st.form_submit_button("Train & promote", type="primary", disabled=running)
    if submitted:
        params = TrainParams(sample_size=None if sample >= max_size else int(sample), n_estimators=int(rounds),
                             max_depth=int(depth), learning_rate=float(lr), subsample=float(sub), seed=int(seed))
        st.session_state["retrain_running"] = True
        try:
            with st.status("Training…", expanded=True) as status:
                version = run_training(params, trips=trips, store=state.artifact_store(),
                                       progress=lambda m: st.write(m))
                status.update(label=f"Promoted {version}", state="complete")
        except Exception as exc:
            st.error(f"Training failed; the previous model version is still live. Details: {exc}")
            return
        finally:
            st.session_state["retrain_running"] = False
        state.clear_caches()
        st.session_state["celebrate"] = version
        st.rerun()


@st.dialog("CRISP-DM research report", width="large")
def report_dialog(bundle, session) -> None:
    content = build_report(bundle.metadata if bundle else None, bundle.experiments if bundle else None,
                           bundle.diagnostics if bundle else None, bundle.holdout if bundle else None,
                           session, state.data_report())
    st.caption("Generated from the active model version, its data report and the latest AutoResearch session"
               + (f" · version `{bundle.version}`" if bundle else " · no model trained yet"))
    nav, body = st.columns([1, 3], gap="medium")
    with nav:
        phase = st.radio("Phase", [p for p, _ in PHASES], format_func=dict(PHASES).get, key="report_phase",
                         label_visibility="collapsed")
    with body:
        st.markdown(f"### {dict(PHASES)[phase]}")
        for kind, value in content[phase]:
            if kind == "md":
                st.markdown(value)
            elif kind == "latex":
                st.latex(value)
            elif kind == "table" and value:
                st.dataframe(value, hide_index=True, width="stretch")
