"""F03/F04/S07 - survival classification workspace and the live predictor."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.components import glossary_expander, metric_tile, workspace_error
from skills_lab.benchmarks.classification import predict_profile
from skills_lab.catalog import View
from skills_lab.config import CLASSIFICATION_TEST_SIZE, SEED


def render(lab) -> None:
    result = lab.result(View.CLASSIFICATION)
    if result is None:
        workspace_error(lab.error(View.CLASSIFICATION) or "Result unavailable.")
        return

    st.subheader("Survival classification - Titanic")
    st.caption(
        f"{result.dataset}. {result.rows:,} passengers, {result.positive_rate * 100:.1f}% "
        f"survived. Stratified hold-out ({int(CLASSIFICATION_TEST_SIZE * 100)}% test, "
        f"seed {SEED}); {result.age_missing_rate * 100:.1f}% of ages are missing in the "
        "source data and are imputed with the training median inside the pipeline."
    )

    tiles = st.columns(3)
    with tiles[0]:
        metric_tile("ROC-AUC", f"{result.roc_auc:.3f}")
    with tiles[1]:
        metric_tile("F1", f"{result.f1:.3f}")
    with tiles[2]:
        metric_tile("Accuracy", f"{result.accuracy * 100:.1f}%")

    _predictor(result)
    st.divider()

    left, right = st.columns([1, 1])
    with left:
        _confusion(result)
    with right:
        _importances(result)

    _roc(result)
    _samples(result)
    glossary_expander(
        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Leakage", "Stratified split"]
    )


def _predictor(result) -> None:
    st.markdown("#### Live predictor")
    st.caption(
        "These controls are scored by the fitted gradient-boosting pipeline itself - the "
        "same object evaluated above. Family size and the travelling-alone flag are "
        "recomputed from the sibling/spouse and parent/child counts exactly as they are in "
        "the training data."
    )
    controls, output = st.columns([3, 2])
    with controls:
        row1 = st.columns(3)
        sex = row1[0].selectbox("Sex", ["female", "male"], key="pred_sex")
        pclass = row1[1].selectbox("Ticket class", [1, 2, 3], key="pred_class")
        embarked = row1[2].selectbox(
            "Port of embarkation", ["S", "C", "Q"], key="pred_port"
        )
        row2 = st.columns(2)
        age = row2[0].slider("Age", 1, 80, 30, key="pred_age")
        fare = row2[1].slider("Fare paid", 5, 300, 50, key="pred_fare")
        row3 = st.columns(2)
        sibsp = row3[0].slider("Siblings / spouse aboard", 0, 8, 0, key="pred_sibsp")
        parch = row3[1].slider("Parents / children aboard", 0, 6, 0, key="pred_parch")

    probability = predict_profile(
        result.pipeline,
        pclass=pclass,
        sex=sex,
        age=age,
        fare=fare,
        sibsp=sibsp,
        parch=parch,
        embarked=embarked,
    )
    with output:
        survived = probability >= 0.5
        st.metric("Predicted survival probability", f"{probability * 100:.1f}%")
        if survived:
            st.success("At a 0.50 cutoff this passenger is classified as **survived**.")
        else:
            st.error("At a 0.50 cutoff this passenger is classified as **did not survive**.")
        st.caption(
            f"Derived features sent to the model: family size {sibsp + parch + 1}, "
            f"{'travelling alone' if sibsp + parch == 0 else 'with family'}. "
            "A tree ensemble need not respond to every field at every profile, so some "
            "single-step changes leave the probability unchanged."
        )


def _confusion(result) -> None:
    st.markdown("#### Confusion matrix (held-out passengers)")
    c = result.confusion
    grid = st.columns(2)
    grid[0].metric("True positives - survived, predicted survived", c["true_positive"])
    grid[1].metric("False positives - died, predicted survived", c["false_positive"])
    grid2 = st.columns(2)
    grid2[0].metric("False negatives - survived, predicted died", c["false_negative"])
    grid2[1].metric("True negatives - died, predicted died", c["true_negative"])
    st.caption(
        f"Precision {result.precision:.3f} - of those predicted to survive, that share did. "
        f"Recall {result.recall:.3f} - of those who survived, that share was found."
    )


def _importances(result) -> None:
    st.markdown("#### Feature importance")
    frame = pd.DataFrame(
        [{"feature": i.feature, "share": i.share_pct} for i in result.importances]
    )
    figure = px.bar(
        frame.sort_values("share"),
        x="share",
        y="feature",
        orientation="h",
        labels={"share": "share of total importance (%)", "feature": ""},
    )
    figure.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Bars are each feature's share of total split importance, on a common axis. "
        "Importance describes what the model used, not what caused survival."
    )


def _roc(result) -> None:
    with st.expander("ROC curve"):
        frame = pd.DataFrame(result.roc_curve)
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(x=frame["x"], y=frame["y"], mode="lines", name="model")
        )
        figure.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="random",
                line=dict(dash="dash", color="grey"),
            )
        )
        figure.update_layout(
            xaxis_title="false-positive rate",
            yaxis_title="true-positive rate",
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(figure, width="stretch")


def _samples(result) -> None:
    with st.expander("Sample held-out rows"):
        st.caption("Missing values are shown as 'missing' rather than silently filled.")
        # Rendered as text: these preview columns deliberately mix numbers with the
        # explicit 'missing' marker.
        st.dataframe(pd.DataFrame(result.sample_rows).astype(str), width="stretch")
