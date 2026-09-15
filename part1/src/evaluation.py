"""Cross-validation and held-out test evaluation for the model pipelines.

`class` is mapped so poisonous = 1 (the positive class), so precision/recall/F1
below are with respect to *predicting poisonous*. A false negative here means
a poisonous mushroom predicted edible - the costly mistake this task cares
about most.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_SEED = 42
CV_FOLDS = 5
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Sequential single-hue ramp (blue), consistent with the rest of the notebook.
HEATMAP_CMAP = "Blues"


def cross_validate_models(models: dict, X, y, cv_folds: int = CV_FOLDS) -> pd.DataFrame:
    """Stratified k-fold CV for each model, on the training set only.

    Each pipeline is cloned fresh per fold by `cross_validate`, so encoding is
    refit on each fold's training portion - no leakage between folds.
    Returns one row per model with the mean and std of each metric.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def evaluate_on_test(models: dict, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Fitted-model metrics on the held-out test set, plus each confusion matrix.

    Returns (metrics_table, confusion_matrices), where confusion_matrices maps
    model name -> (tn, fp, fn, tp).
    """
    rows = []
    confusion_matrices = {}
    for name, pipeline in models.items():
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        confusion_matrices[name] = (tn, fp, fn, tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({
            "model": name,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "false_negatives": fn,
            "false_positives": fp,
        })
    return pd.DataFrame(rows).set_index("model"), confusion_matrices


def plot_confusion_matrices(confusion_matrices: dict, n_cols: int = 3):
    """Small-multiples grid of confusion matrix heatmaps, one per model."""
    class_names = ["Edible", "Poisonous"]
    n_models = len(confusion_matrices)
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)

    for ax, (name, (tn, fp, fn, tp)) in zip(axes, confusion_matrices.items()):
        matrix = np.array([[tn, fp], [fn, tp]])
        ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0)
        for i in range(2):
            for j in range(2):
                value = matrix[i, j]
                color = "white" if value > matrix.max() * 0.6 else "#0b0b0b"
                ax.text(j, i, f"{value}", ha="center", va="center", color=color)
        ax.set_xticks([0, 1], class_names)
        ax.set_yticks([0, 1], class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=10)

    for ax in axes[n_models:]:
        ax.axis("off")

    fig.tight_layout()
    return fig
