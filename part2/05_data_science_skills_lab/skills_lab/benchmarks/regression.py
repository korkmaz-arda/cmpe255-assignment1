"""F05 - Ames house prices: random forest on a log1p target, scored in dollars.

The model is fitted against log1p(SalePrice) because sale prices are right-skewed;
predictions are back-transformed with expm1 before any metric is computed, so RMSE, MAE
and R-squared are all in original currency units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skills_lab.config import REGRESSION_TEST_SIZE, RF_PARAMS, SEED
from skills_lab.data import ames
from skills_lab.benchmarks.common import (
    Importance,
    feature_names,
    make_preprocessor,
    sample_rows,
    top_importances,
)


@dataclass
class RegressionResult:
    dataset: str
    rows: int
    train_rows: int
    test_rows: int
    target_transform: str
    rmse: float
    mae: float
    r2: float
    predictions: list[dict[str, float]]
    importances: list[Importance]
    sample_rows: list[dict]
    price_median: float
    area_basement_correlation: float
    pipeline: Pipeline | None = field(default=None, repr=False)

    def payload(self) -> dict:
        return {
            "dataset": self.dataset,
            "rows": self.rows,
            "train_rows": self.train_rows,
            "test_rows": self.test_rows,
            "target_transform": self.target_transform,
            "metrics": {
                "rmse_usd": round(self.rmse, 2),
                "mae_usd": round(self.mae, 2),
                "r2": round(self.r2, 4),
            },
            "top_features": [
                {"feature": i.feature, "share_pct": round(i.share_pct, 2)}
                for i in self.importances
            ],
            "sample_rows": self.sample_rows,
            "notes": {
                "split": f"random hold-out, test_size={REGRESSION_TEST_SIZE}, seed={SEED}",
                "scoring": "metrics computed after expm1 back-transformation, in USD",
                "living_area_vs_basement_correlation": round(
                    self.area_basement_correlation, 3
                ),
            },
        }


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(ames.NUMERIC_FEATURES, ames.CATEGORICAL_FEATURES),
            ),
            ("model", RandomForestRegressor(**RF_PARAMS)),
        ]
    )


def run(frame: pd.DataFrame | None = None) -> RegressionResult:
    data = ames.load() if frame is None else frame
    X = data[ames.FEATURES]
    y = data[ames.TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=REGRESSION_TEST_SIZE, random_state=SEED
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, np.log1p(y_train))

    predicted = np.expm1(pipeline.predict(X_test))
    actual = y_test.to_numpy(dtype=float)

    names = feature_names(pipeline.named_steps["preprocess"])
    importances = top_importances(
        names, pipeline.named_steps["model"].feature_importances_
    )

    points = [
        {
            "actual": float(a),
            "predicted": float(p),
            "residual": float(a - p),
        }
        for a, p in list(zip(actual, predicted))[:200]
    ]

    return RegressionResult(
        dataset="Ames, Iowa house sales (OpenML 42165)",
        rows=int(len(data)),
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        target_transform="log1p on fit, expm1 before scoring",
        rmse=float(np.sqrt(mean_squared_error(actual, predicted))),
        mae=float(mean_absolute_error(actual, predicted)),
        r2=float(r2_score(actual, predicted)),
        predictions=points,
        importances=importances,
        sample_rows=sample_rows(X_test.assign(SalePrice=y_test)),
        price_median=float(data[ames.TARGET].median()),
        area_basement_correlation=float(
            data["GrLivArea"].corr(data["TotalBsmtSF"])
        ),
        pipeline=pipeline,
    )
