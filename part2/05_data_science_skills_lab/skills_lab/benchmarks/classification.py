"""F03/F04 - Titanic survival: gradient-boosted classification on a stratified hold-out.

Returns both the metrics and the fitted pipeline, because the live predictor (F04) scores
user-entered passenger profiles with this same fitted pipeline rather than a stand-in
formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skills_lab.config import CLASSIFICATION_TEST_SIZE, GB_PARAMS, SEED
from skills_lab.data import titanic
from skills_lab.benchmarks.common import (
    Importance,
    downsample_curve,
    feature_names,
    make_preprocessor,
    sample_rows,
    top_importances,
)


@dataclass
class ClassificationResult:
    dataset: str
    rows: int
    train_rows: int
    test_rows: int
    positive_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: dict[str, int]
    roc_curve: list[dict[str, float]]
    importances: list[Importance]
    sample_rows: list[dict]
    age_missing_rate: float
    pipeline: Pipeline | None = field(default=None, repr=False)

    def payload(self) -> dict:
        """JSON-safe view used by the skill-execution modal."""
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "positive_rate": round(self.positive_rate, 4),
            "metrics": {
                "accuracy": round(self.accuracy, 4),
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
                "roc_auc": round(self.roc_auc, 4),
            },
            "confusion_matrix": self.confusion,
            "top_features": [
                {"feature": i.feature, "share_pct": round(i.share_pct, 2)}
                for i in self.importances
            ],
            "sample_rows": self.sample_rows,
            "notes": {
                "split": f"stratified hold-out, test_size={CLASSIFICATION_TEST_SIZE}, seed={SEED}",
                "preprocessing": "median impute + standardise numerics, one-hot categoricals, fitted on train only",
                "age_missing_rate": round(self.age_missing_rate, 4),
            },
        }


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(titanic.NUMERIC_FEATURES, titanic.CATEGORICAL_FEATURES),
            ),
            ("model", GradientBoostingClassifier(**GB_PARAMS)),
        ]
    )


def run(frame: pd.DataFrame | None = None) -> ClassificationResult:
    data = titanic.load() if frame is None else frame
    X = data[titanic.FEATURES]
    y = data[titanic.TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CLASSIFICATION_TEST_SIZE, random_state=SEED, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_test, proba)

    names = feature_names(pipeline.named_steps["preprocess"])
    importances = top_importances(
        names, pipeline.named_steps["model"].feature_importances_
    )

    return ClassificationResult(
        dataset="Titanic passenger list (OpenML 40945)",
        rows=int(len(data)),
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        positive_rate=float(y.mean()),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        confusion={
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        roc_curve=downsample_curve(fpr, tpr),
        importances=importances,
        sample_rows=sample_rows(X_test.assign(survived=y_test)),
        age_missing_rate=float(data["age"].isna().mean()),
        pipeline=pipeline,
    )


def predict_profile(
    pipeline: Pipeline,
    *,
    pclass: int,
    sex: str,
    age: float,
    fare: float,
    sibsp: int,
    parch: int,
    embarked: str,
) -> float:
    """Survival probability for one user-entered profile, from the fitted pipeline.

    The two engineered features are recomputed here exactly as they are in the training
    data, so the controls the UI exposes reach the model through the same derivation.
    """
    row = pd.DataFrame(
        [
            {
                "pclass": str(int(pclass)),
                "sex": str(sex),
                "age": float(age),
                "fare": float(fare),
                "sibsp": int(sibsp),
                "parch": int(parch),
                "embarked": str(embarked),
            }
        ]
    )
    row = titanic.add_derived_features(row)
    return float(pipeline.predict_proba(row[titanic.FEATURES])[0, 1])
