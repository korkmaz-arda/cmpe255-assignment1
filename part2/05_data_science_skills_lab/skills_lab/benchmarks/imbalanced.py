"""F06/F07 - credit-card fraud: unweighted vs class-weighted logistic regression.

Two disciplines are enforced here.

1. **Primary metric.** At a 0.17% base rate, ROC-AUC is easy to make look excellent, so
   Average Precision (the area under the precision-recall curve) is the headline metric and
   ROC-AUC is reported as secondary context.

2. **Thresholds live on the score scale.** Class weighting deliberately distorts the
   probability scale: the weighted model pushes scores so hard that thousands of rows round
   to a predicted probability of exactly 1.0 in float64, leaving no usable resolution near
   the top. All cutoff logic therefore runs on the model's decision function (log-odds),
   where every distinct decision is representable. The default cutoff (log-odds 0) is
   exactly the familiar probability 0.5.

3. **The test partition is never used to choose anything.** The data is split three ways.
   Both models are fitted on train. Every threshold decision - the sweep, the F1 optimum,
   and the interactive slider - reads *validation* scores only. The chosen threshold is
   then applied once to the held-out test partition and reported separately as the final
   evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skills_lab.config import FRAUD_TEST_SIZE, FRAUD_VALIDATION_SIZE, LOGREG_PARAMS, SEED
from skills_lab.data import fraud
from skills_lab.benchmarks.common import downsample_curve

DEFAULT_THRESHOLD = 0.0  # log-odds 0 == probability 0.5


@dataclass
class ModelScores:
    name: str
    class_weight: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    roc_auc: float
    confusion: dict[str, int]

    def as_dict(self) -> dict:
        return {
            "model": self.name,
            "class_weight": self.class_weight,
            "threshold": round(self.threshold, 6),
            "accuracy": round(self.accuracy, 5),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "average_precision": round(self.average_precision, 4),
            "roc_auc": round(self.roc_auc, 4),
            "confusion_matrix": self.confusion,
        }


@dataclass
class ImbalancedResult:
    dataset: str
    rows: int
    positives: int
    positive_rate: float
    train_rows: int
    validation_rows: int
    test_rows: int
    baseline_test: ModelScores
    weighted_test: ModelScores
    selected_threshold: float
    selected_validation_f1: float
    weighted_selected_test: ModelScores
    pr_curve_validation: list[dict[str, float]]
    threshold_sweep: list[dict[str, float]]
    validation_scores: np.ndarray | None = field(default=None, repr=False)
    validation_labels: np.ndarray | None = field(default=None, repr=False)

    def payload(self) -> dict:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "positives": self.positives,
            "positive_rate": round(self.positive_rate, 6),
            "split": {
                "train": self.train_rows,
                "validation": self.validation_rows,
                "test": self.test_rows,
                "note": "stratified; thresholds are chosen on validation only",
            },
            "primary_metric": "average_precision (PR-AUC)",
            "baseline_at_default_cutoff_test": self.baseline_test.as_dict(),
            "class_weighted_at_default_cutoff_test": self.weighted_test.as_dict(),
            "threshold_selection": {
                "selected_threshold": round(self.selected_threshold, 6),
                "selection_partition": "validation",
                "validation_f1_at_selected": round(self.selected_validation_f1, 4),
                "candidates": len(self.threshold_sweep),
                "candidate_source": "every threshold on the validation precision-recall curve",
                "threshold_scale": "model decision function (log-odds); 0 == probability 0.5",
            },
            "final_held_out_evaluation": self.weighted_selected_test.as_dict(),
        }


def build_pipeline(class_weight: str | None) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(class_weight=class_weight, **LOGREG_PARAMS),
            ),
        ]
    )


def _score(
    name: str,
    class_weight: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> ModelScores:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return ModelScores(
        name=name,
        class_weight=class_weight,
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, pred)),
        precision=float(precision_score(y_true, pred, zero_division=0)),
        recall=float(recall_score(y_true, pred, zero_division=0)),
        f1=float(f1_score(y_true, pred, zero_division=0)),
        average_precision=float(average_precision_score(y_true, scores)),
        roc_auc=float(roc_auc_score(y_true, scores)),
        confusion={
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    )


def threshold_metrics(
    y_true: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Measured precision/recall/F1 at one cutoff. Used by the interactive workbench."""
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "flagged": int(tp + fp),
        "caught": int(tp),
        "missed": int(fn),
        "false_alarms": int(fp),
    }


def _sweep_from_pr_curve(
    y_val: np.ndarray, scores_val: np.ndarray
) -> tuple[list[dict[str, float]], float, float]:
    """Candidate thresholds are the validation PR-curve thresholds themselves.

    A fixed grid such as 30 points over 0.10-0.90 cannot find the optimum at this base
    rate; the PR curve gives every cutoff at which the confusion matrix actually changes.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, scores_val)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))
    sweep = [
        {
            "threshold": float(t),
            "precision": float(p),
            "recall": float(r),
            "f1": float(s),
        }
        for t, p, r, s in zip(thresholds, precision, recall, f1)
    ]
    return sweep, float(thresholds[best]), float(f1[best])


def run(frame: pd.DataFrame | None = None) -> ImbalancedResult:
    data = fraud.load() if frame is None else frame
    X = data[fraud.FEATURES].to_numpy(dtype=float)
    y = data[fraud.TARGET].to_numpy(dtype=int)

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=FRAUD_TEST_SIZE, random_state=SEED, stratify=y
    )
    validation_share = FRAUD_VALIDATION_SIZE / (1 - FRAUD_TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=validation_share, random_state=SEED, stratify=y_rest
    )

    baseline = build_pipeline(None).fit(X_train, y_train)
    weighted = build_pipeline("balanced").fit(X_train, y_train)

    baseline_test_scores = baseline.decision_function(X_test)
    weighted_test_scores = weighted.decision_function(X_test)
    weighted_val_scores = weighted.decision_function(X_val)

    sweep, selected_threshold, selected_val_f1 = _sweep_from_pr_curve(
        y_val, weighted_val_scores
    )
    precision_v, recall_v, _ = precision_recall_curve(y_val, weighted_val_scores)

    return ImbalancedResult(
        dataset="ULB credit-card fraud (OpenML 1597)",
        rows=int(len(data)),
        positives=int(y.sum()),
        positive_rate=float(y.mean()),
        train_rows=int(len(y_train)),
        validation_rows=int(len(y_val)),
        test_rows=int(len(y_test)),
        baseline_test=_score(
            "Unweighted logistic regression", "none", y_test, baseline_test_scores,
            DEFAULT_THRESHOLD,
        ),
        weighted_test=_score(
            "Class-weighted logistic regression", "balanced", y_test,
            weighted_test_scores, DEFAULT_THRESHOLD,
        ),
        selected_threshold=selected_threshold,
        selected_validation_f1=selected_val_f1,
        weighted_selected_test=_score(
            "Class-weighted logistic regression @ selected threshold", "balanced",
            y_test, weighted_test_scores, selected_threshold,
        ),
        pr_curve_validation=downsample_curve(recall_v, precision_v),
        threshold_sweep=sweep,
        validation_scores=weighted_val_scores,
        validation_labels=y_val,
    )


def review_rate_for_threshold(scores_val: np.ndarray, threshold: float) -> float:
    """Share of validation transactions a given threshold would flag."""
    return float((scores_val >= threshold).mean())


def review_budget_thresholds(
    scores_val: np.ndarray, points: int = 60, include: float | None = None
) -> list[dict[str, float]]:
    """Decision thresholds expressed as the share of transactions sent for review.

    The class-weighted model pushes scores towards 1, so its useful cutoffs live in a
    narrow band near the top of the range and a plain 0-1 slider has no usable
    resolution there. Parameterising the same decision threshold by review budget - the
    fraction of transactions flagged - is a monotone relabelling that stays legible:
    a larger budget means a lower threshold, more recall and less precision.
    """
    rates = np.logspace(np.log10(0.0001), np.log10(0.05), points)
    budgets = [
        {"review_rate": float(rate), "threshold": float(np.quantile(scores_val, 1 - rate))}
        for rate in rates
    ]
    if include is not None:
        # Keep the F1-optimal threshold itself on the slider, so its default position
        # reports exactly the selected cutoff rather than the nearest grid point.
        budgets.append(
            {
                "review_rate": review_rate_for_threshold(scores_val, include),
                "threshold": float(include),
            }
        )
        budgets.sort(key=lambda item: item["review_rate"])
    return budgets


def as_probability(score: float) -> float:
    """Convert a log-odds cutoff back to the model's probability scale for display."""
    return float(1.0 / (1.0 + np.exp(-score)))
