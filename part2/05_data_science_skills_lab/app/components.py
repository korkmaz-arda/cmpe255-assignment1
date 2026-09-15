"""Small shared UI pieces: metric tiles with plain-language help, badges, and glossary."""

from __future__ import annotations

import streamlit as st

GLOSSARY = {
    "Accuracy": "Share of all predictions that were right. Misleading when one class is rare.",
    "Precision": "Of the cases the model flagged, the share that really were positive.",
    "Recall": "Of the cases that really were positive, the share the model flagged.",
    "F1": "The harmonic mean of precision and recall - high only when both are high.",
    "ROC-AUC": "Probability that a random positive is ranked above a random negative. "
               "Stays high on very imbalanced data, so treat it as context, not a headline.",
    "Average precision (PR-AUC)": "Area under the precision-recall curve. The right "
        "threshold-free summary when positives are rare, because precision's denominator "
        "is the alerts you actually raise.",
    "RMSE": "Root mean squared error: typical error size in the target's own units, with "
            "big misses weighted more heavily.",
    "MAE": "Mean absolute error: the average miss, in the target's own units.",
    "R-squared": "Share of the variance in the target the model explains; 1.0 is perfect, "
                 "0 is no better than always predicting the mean.",
    "Decision threshold": "The score above which a case is flagged. Moving it trades "
                          "recall against precision; it is a business choice, not a model parameter.",
    "Log-odds": "The model's raw score. 0 is the familiar probability 0.5; larger means "
                "more confident. Thresholds are set here because probabilities saturate.",
    "Leakage": "Letting information from the evaluation data influence training - it "
               "inflates scores in a way you cannot detect afterwards.",
    "Stratified split": "A split that keeps the class proportions of the original data in "
                        "both partitions.",
    "Cohort": "A group of customers defined by when they were acquired.",
    "Retention": "Share of a cohort that purchased again in a later period. The denominator "
                 "is the cohort, not the whole customer base.",
    "z-score": "How many standard errors apart the two conversion rates are.",
    "p-value": "Probability of seeing a difference at least this large if the variants were "
               "truly identical. It measures distinguishability, not effect size.",
    "Confidence interval": "The range of differences consistent with the data at 95%. "
                           "Its width is the honest statement of how much you don't know.",
    "Percentage point (pp)": "The arithmetic difference between two percentages: 5% to 6% "
                             "is +1 pp, which is a +20% relative lift.",
    "Completeness": "Share of cells that are populated.",
    "Validity": "Share of populated cells that satisfy their column's rule.",
    "Uniqueness": "Share of rows that are not exact duplicates.",
    "Consistency": "Share of text values already written in canonical form.",
}


def metric_tile(label: str, value: str, help_key: str | None = None, delta: str | None = None):
    st.metric(label, value, delta=delta, help=GLOSSARY.get(help_key or label))


def health_badge(state: str) -> str:
    colors = {"PASS": "green", "WARNING": "orange", "CRITICAL": "red"}
    return f":{colors.get(state, 'grey')}[{state}]"


def glossary_expander(keys: list[str] | None = None) -> None:
    items = keys or list(GLOSSARY)
    with st.expander("Glossary - what these terms mean"):
        for key in items:
            if key in GLOSSARY:
                st.markdown(f"**{key}** - {GLOSSARY[key]}")


def workspace_error(message: str) -> None:
    """Honest failure state. This project never falls back to stand-in numbers."""
    st.error(
        "This workspace could not be computed, so no figures are shown.\n\n"
        f"```\n{message}\n```"
    )


def source_note(text: str) -> None:
    st.caption(text)
