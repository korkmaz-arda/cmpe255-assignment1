"""F06/F07/S07 - imbalanced classification: comparison and threshold workbench."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.components import glossary_expander, metric_tile, workspace_error
from skills_lab.benchmarks.imbalanced import (
    as_probability,
    review_budget_thresholds,
    review_rate_for_threshold,
    threshold_metrics,
)
from skills_lab.catalog import View


def render(lab) -> None:
    result = lab.result(View.FRAUD)
    if result is None:
        workspace_error(lab.error(View.FRAUD) or "Result unavailable.")
        return

    st.subheader("Imbalanced classification - credit-card fraud")
    st.caption(
        f"{result.dataset}. {result.rows:,} transactions with {result.positives} frauds - a "
        f"{result.positive_rate * 100:.3f}% base rate, so always predicting 'legitimate' "
        f"would score {100 - result.positive_rate * 100:.3f}% accuracy. Split "
        f"{result.train_rows:,} train / {result.validation_rows:,} validation / "
        f"{result.test_rows:,} test, stratified."
    )

    tiles = st.columns(3)
    with tiles[0]:
        metric_tile(
            "Average precision (PR-AUC)",
            f"{result.weighted_test.average_precision:.3f}",
            help_key="Average precision (PR-AUC)",
            delta=f"{result.weighted_test.average_precision - result.baseline_test.average_precision:+.3f} vs unweighted",
        )
    with tiles[1]:
        metric_tile(
            "Class-weighted recall at default cutoff",
            f"{result.weighted_test.recall * 100:.0f}%",
            help_key="Recall",
        )
    with tiles[2]:
        metric_tile(
            "F1 at the chosen cutoff", f"{result.weighted_selected_test.f1:.3f}"
        )
    st.caption(
        "Average precision is the headline metric here, not accuracy and not ROC-AUC: with "
        "0.17% positives both of those stay flattering even for a weak detector."
    )

    _comparison(result)
    st.divider()
    _workbench(result)
    st.divider()
    _final(result)
    _pr_curve(result)
    glossary_expander(
        [
            "Average precision (PR-AUC)", "ROC-AUC", "Precision", "Recall", "F1",
            "Decision threshold", "Log-odds",
        ]
    )


def _comparison(result) -> None:
    st.markdown("#### Same data, same cutoff, one difference: class weighting")
    rows = []
    for scores, note in (
        (
            result.baseline_test,
            f"misses {scores_missed(result.baseline_test)} of "
            f"{scores_total(result.baseline_test)} frauds",
        ),
        (
            result.weighted_test,
            f"catches {result.weighted_test.confusion['true_positive']} of "
            f"{scores_total(result.weighted_test)} frauds, but raises "
            f"{result.weighted_test.confusion['false_positive']:,} false alarms",
        ),
    ):
        rows.append(
            {
                "Model": scores.name,
                "Class weight": scores.class_weight,
                "Accuracy": f"{scores.accuracy * 100:.2f}%",
                "Precision": f"{scores.precision:.3f}",
                "Recall": f"{scores.recall:.3f}",
                "F1": f"{scores.f1:.3f}",
                "Avg precision": f"{scores.average_precision:.3f}",
                "ROC-AUC": f"{scores.roc_auc:.3f}",
                "Operational reading": note,
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    baseline_fp = result.baseline_test.confusion["false_positive"]
    weighted_fp = result.weighted_test.confusion["false_positive"]
    st.caption(
        f"Both rows are measured on the held-out test partition at the default cutoff "
        f"(log-odds 0, i.e. probability 0.5). Accuracy stays high in both rows "
        f"({result.baseline_test.accuracy * 100:.2f}% then "
        f"{result.weighted_test.accuracy * 100:.2f}%), which is exactly why it is the wrong "
        f"headline: that {abs(result.weighted_test.accuracy - result.baseline_test.accuracy) * 100:.2f} "
        f"percentage-point move hides recall rising from "
        f"{result.baseline_test.recall * 100:.0f}% to {result.weighted_test.recall * 100:.0f}% "
        f"and false alarms rising from {baseline_fp:,} to {weighted_fp:,} "
        f"({weighted_fp / max(baseline_fp, 1):.0f}x). Neither row is deployable as-is: "
        f"weighting buys recall at a precision cost that only a deliberate threshold choice "
        f"can manage."
    )


def scores_total(scores) -> int:
    return scores.confusion["true_positive"] + scores.confusion["false_negative"]


def scores_missed(scores) -> int:
    return scores.confusion["false_negative"]


def _workbench(result) -> None:
    st.markdown("#### Decision-threshold workbench")
    st.caption(
        "Measured on the **validation** partition, so exploring here cannot contaminate the "
        "final held-out numbers below. The control is the review budget - the share of "
        "transactions sent for manual review - which is a direct relabelling of the decision "
        "threshold: a bigger budget means a lower threshold, more recall, less precision."
    )

    budgets = review_budget_thresholds(
        result.validation_scores, include=result.selected_threshold
    )
    options = [b["review_rate"] for b in budgets]
    thresholds = {b["review_rate"]: b["threshold"] for b in budgets}

    selected_rate = review_rate_for_threshold(
        result.validation_scores, result.selected_threshold
    )
    default = selected_rate if selected_rate in thresholds else min(
        options, key=lambda r: abs(r - selected_rate)
    )

    st.caption(
        "The slider starts at the F1-optimal cutoff chosen on validation, so the three "
        "tiles below open showing that selected operating point."
    )
    rate = st.select_slider(
        "Review budget (share of transactions flagged)",
        options=options,
        value=default,
        format_func=lambda r: f"{r * 100:.3f}%",
        key="fraud_budget",
    )
    threshold = thresholds[rate]
    metrics = threshold_metrics(
        result.validation_labels, result.validation_scores, threshold
    )

    tiles = st.columns(4)
    tiles[0].metric("Recall", f"{metrics['recall'] * 100:.1f}%")
    tiles[1].metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    tiles[2].metric("F1", f"{metrics['f1']:.3f}")
    tiles[3].metric("Decision threshold (log-odds)", f"{threshold:.2f}")
    st.caption(
        f"At this cutoff the model flags {metrics['flagged']:,} validation transactions, "
        f"catching {metrics['caught']} frauds, missing {metrics['missed']}, and raising "
        f"{metrics['false_alarms']:,} false alarms. Equivalent probability cutoff: "
        f"{as_probability(threshold):.6f} - probabilities saturate near 1 after class "
        "weighting, which is why the threshold is set on the log-odds scale."
    )

    sweep = pd.DataFrame(result.threshold_sweep)
    sweep = sweep[sweep["threshold"].between(-5, 30)]
    figure = px.line(
        sweep.melt("threshold", ["precision", "recall", "f1"], var_name="metric"),
        x="threshold",
        y="value",
        color="metric",
        labels={"threshold": "decision threshold (log-odds)", "value": ""},
    )
    figure.add_vline(x=threshold, line_dash="dot")
    figure.add_vline(x=result.selected_threshold, line_dash="dash", line_color="grey")
    figure.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Dotted line: your current cutoff. Dashed line: the F1-optimal cutoff selected on "
        "validation. Curves are measured on validation at every cutoff where the confusion "
        "matrix changes."
    )


def _final(result) -> None:
    st.markdown("#### Final held-out evaluation (test partition, scored once)")
    st.caption(
        f"The production cutoff was chosen on validation (log-odds "
        f"{result.selected_threshold:.2f}, validation F1 "
        f"{result.selected_validation_f1:.3f}) from every threshold on the validation "
        "precision-recall curve - not from a fixed grid, because at this base rate the "
        "optimum does not sit in a convenient 0.1-0.9 window. That single fixed rule is then "
        "applied once to the test partition."
    )
    final = result.weighted_selected_test
    tiles = st.columns(4)
    tiles[0].metric("Test recall", f"{final.recall * 100:.1f}%")
    tiles[1].metric("Test precision", f"{final.precision * 100:.1f}%")
    tiles[2].metric("Test F1", f"{final.f1:.3f}")
    tiles[3].metric(
        "Validation → test F1 gap",
        f"{final.f1 - result.selected_validation_f1:+.3f}",
    )
    st.info(
        f"On test the rule catches {final.confusion['true_positive']} of "
        f"{scores_total(final)} frauds with {final.confusion['false_positive']:,} false "
        f"alarms. Reporting the validation F1 as the result would have overstated it: the "
        f"gap indicates how much of that validation score came from fitting the cutoff to "
        f"validation data (ordinary sampling differences between the two partitions "
        f"contribute as well). Selecting the cutoff on test - as the specification this "
        f"project reimplements did - hides that gap entirely."
    )


def _pr_curve(result) -> None:
    with st.expander("Precision-recall curve (validation)"):
        frame = pd.DataFrame(result.pr_curve_validation).rename(
            columns={"x": "recall", "y": "precision"}
        )
        figure = px.line(frame, x="recall", y="precision")
        figure.add_hline(
            y=result.positive_rate, line_dash="dash", line_color="grey",
            annotation_text="random classifier (base rate)",
        )
        figure.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figure, width="stretch")
