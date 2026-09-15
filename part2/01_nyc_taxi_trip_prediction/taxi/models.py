"""Model families, target transform, and metrics (F04)."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config


def to_log_target(seconds) -> np.ndarray:
    return np.log1p(np.asarray(seconds, dtype=float))


def from_log_target(log_pred) -> np.ndarray:
    return np.expm1(np.asarray(log_pred, dtype=float))


def regression_metrics(y_log_true, y_log_pred) -> dict[str, float]:
    """RMSLE in log1p space; RMSE / MAE / R^2 in seconds after inverting the transform."""
    y_log_true = np.asarray(y_log_true, dtype=float)
    y_log_pred = np.asarray(y_log_pred, dtype=float)
    sec_true = from_log_target(y_log_true)
    sec_pred = from_log_target(y_log_pred)
    return {
        "rmsle": float(np.sqrt(mean_squared_error(y_log_true, y_log_pred))),
        "rmse_s": float(np.sqrt(mean_squared_error(sec_true, sec_pred))),
        "mae_s": float(mean_absolute_error(sec_true, sec_pred)),
        "r2": float(r2_score(sec_true, sec_pred)),
        "n": int(len(y_log_true)),
    }


class MedianBaseline:
    """Naive reference: predicts the training median of log1p(duration) for every trip."""

    def fit(self, X, y):
        self.value_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value_)


@dataclass
class FittedModel:
    key: str
    model: Any
    fit_seconds: float
    hyperparameters: dict
    curves: dict | None = None
    best_iteration: int | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if isinstance(self.model, xgb.XGBRegressor) and self.best_iteration is not None:
            return self.model.predict(X, iteration_range=(0, self.best_iteration + 1))
        return self.model.predict(X)


FAMILY = {
    "baseline": "Reference baseline (training median)",
    "ridge": "L2-regularised linear regression",
    "random_forest": "Bagged decision forest",
    "xgboost": "Gradient-boosted trees (XGBoost)",
    "hist_gb": "Histogram gradient boosting (scikit-learn)",
    "extra_trees": "Extremely randomised trees",
    "mlp": "Feed-forward neural network (MLP)",
}
DISPLAY_NAME = {
    "baseline": "Median baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "hist_gb": "HistGradientBoosting",
    "extra_trees": "Extra trees",
    "mlp": "MLP neural network",
}


def default_params(key: str, seed: int) -> dict:
    if key == "baseline":
        return {"statistic": "median of log1p(duration)"}
    if key == "ridge":
        return {"alpha": 1.0, "scaling": "standard"}
    if key == "random_forest":
        return {"n_estimators": 40, "max_depth": 14, "min_samples_leaf": 5, "max_samples": 0.5,
                "random_state": seed}
    if key == "extra_trees":
        return {"n_estimators": 40, "max_depth": 16, "min_samples_leaf": 5, "max_samples": 0.5,
                "bootstrap": True, "random_state": seed}
    if key == "hist_gb":
        return {"max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 63, "early_stopping": False,
                "random_state": seed}
    if key == "mlp":
        return {"hidden_layer_sizes": [64, 32], "alpha": 1e-4, "max_iter": 40, "early_stopping": True,
                "scaling": "standard", "random_state": seed}
    if key == "xgboost":
        return {**config.DEFAULT_XGB, "random_state": seed}
    raise KeyError(key)


def fit_model(key: str, params: dict, X_train: pd.DataFrame, y_train: np.ndarray,
              X_val: pd.DataFrame | None = None, y_val: np.ndarray | None = None,
              curve_rows: int = 100_000) -> FittedModel:
    """Fit one model family. XGBoost early-stops on the provided validation set."""
    params = dict(params)
    t0 = time.perf_counter()
    curves = None
    best_iteration = None
    if key == "baseline":
        model = MedianBaseline().fit(X_train, y_train)
    elif key == "ridge":
        model = make_pipeline(StandardScaler(), Ridge(alpha=params["alpha"])).fit(X_train, y_train)
    elif key in ("random_forest", "extra_trees"):
        cls = RandomForestRegressor if key == "random_forest" else ExtraTreesRegressor
        kw = {k: v for k, v in params.items()}
        model = cls(n_jobs=-1, **kw).fit(X_train, y_train)
    elif key == "hist_gb":
        model = HistGradientBoostingRegressor(**params).fit(X_train, y_train)
    elif key == "mlp":
        kw = {k: v for k, v in params.items() if k != "scaling"}
        kw["hidden_layer_sizes"] = tuple(kw["hidden_layer_sizes"])
        model = make_pipeline(StandardScaler(), MLPRegressor(**kw)).fit(X_train, y_train)
    elif key == "xgboost":
        if X_val is None:
            raise ValueError("XGBoost requires a validation set for early stopping")
        model = xgb.XGBRegressor(
            objective="reg:squarederror", tree_method="hist", eval_metric="rmse", n_jobs=-1,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS, **params,
        )
        # Training-loss curve is tracked on a fixed seeded subset of the training rows for speed;
        # the early-stopping set (last in eval_set) is the full validation split.
        rng = np.random.default_rng(0)
        n_curve = min(curve_rows, len(X_train))
        cidx = np.sort(rng.choice(len(X_train), size=n_curve, replace=False))
        model.fit(X_train, y_train,
                  eval_set=[(X_train.iloc[cidx], y_train[cidx]), (X_val, y_val)], verbose=False)
        ev = model.evals_result()
        curves = {"train_rmse": ev["validation_0"]["rmse"], "validation_rmse": ev["validation_1"]["rmse"],
                  "train_curve_rows": int(n_curve)}
        best_iteration = int(model.best_iteration)
    else:
        raise KeyError(key)
    return FittedModel(key=key, model=model, fit_seconds=time.perf_counter() - t0,
                       hyperparameters=params, curves=curves, best_iteration=best_iteration)


def gain_importance(fitted: FittedModel, feature_names: list[str]) -> list[dict]:
    """XGBoost total-gain importance, normalised to percentages, descending."""
    booster = fitted.model.get_booster()
    scores = booster.get_score(importance_type="total_gain")
    raw = [(f, float(scores.get(f, 0.0))) for f in feature_names]
    total = sum(v for _, v in raw) or 1.0
    ranked = sorted(raw, key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "importance": v, "percent": 100.0 * v / total} for f, v in ranked]
