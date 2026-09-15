"""Full training pipeline: sample → features → benchmark → band → telemetry → version → promote (F04/F05/F13).

Reads only the fit, calibration and validation roles. The test role is never touched here.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from . import config, models
from .artifacts import ArtifactStore, new_version_id, utc_now_iso, write_json_atomic
from .data import TripStore, sample_rows
from .diagnostics import band_coverage, deep_dive, downsample_curve, fit_band, training_telemetry
from .features import BASE_FEATURES, build_features

BENCHMARK_KEYS = ("baseline", "ridge", "random_forest", "xgboost")
PRODUCTION_KEY = "xgboost"


@dataclass
class TrainParams:
    sample_size: int | None = None  # None = whole fit pool
    n_estimators: int = config.DEFAULT_XGB["n_estimators"]
    max_depth: int = config.DEFAULT_XGB["max_depth"]
    learning_rate: float = config.DEFAULT_XGB["learning_rate"]
    subsample: float = config.DEFAULT_XGB["subsample"]
    seed: int = 42

    def validate(self, fit_pool_size: int | None = None) -> None:
        b = config.RETRAIN_BOUNDS
        checks = [
            ("n_estimators", self.n_estimators), ("max_depth", self.max_depth),
            ("learning_rate", self.learning_rate), ("subsample", self.subsample),
        ]
        for name, value in checks:
            lo, hi = b[name]
            if not (lo <= value <= hi):
                raise ValueError(f"{name}={value} outside allowed range [{lo}, {hi}]")
        if self.sample_size is not None:
            lo = b["sample_size"][0]
            if self.sample_size < lo:
                raise ValueError(f"sample_size={self.sample_size} below minimum {lo}")
            if fit_pool_size is not None and self.sample_size > fit_pool_size:
                raise ValueError(f"sample_size={self.sample_size} exceeds fit pool of {fit_pool_size}")

    def xgb_params(self) -> dict:
        return {**config.DEFAULT_XGB, "n_estimators": int(self.n_estimators), "max_depth": int(self.max_depth),
                "learning_rate": float(self.learning_rate), "subsample": float(self.subsample),
                "random_state": int(self.seed)}


def run_training(params: TrainParams, trips: TripStore | None = None, store: ArtifactStore | None = None,
                 progress: Callable[[str], None] | None = None, benchmark_keys=BENCHMARK_KEYS) -> str:
    """Run the full pipeline and promote the new version. Returns the version id.

    On any failure the partially written version is removed and `current.json` is left unchanged.
    """
    trips = trips or TripStore()
    store = store or ArtifactStore()
    say = progress or (lambda msg: None)
    t_start = time.perf_counter()

    say("Loading fit pool, calibration and validation splits")
    fit_pool = trips.role("fit")
    params.validate(len(fit_pool))
    train_df = sample_rows(fit_pool, params.sample_size, params.seed)
    del fit_pool
    calib_df = trips.role("calibration")
    val_df = trips.role("validation")

    say(f"Engineering features for {len(train_df):,} training rows")
    X_tr, y_tr = build_features(train_df), models.to_log_target(train_df[config.TARGET_COLUMN])
    X_cal, y_cal = build_features(calib_df), models.to_log_target(calib_df[config.TARGET_COLUMN])
    X_val, y_val = build_features(val_df), models.to_log_target(val_df[config.TARGET_COLUMN])

    version = new_version_id()
    vdir = store.staging_dir(version)
    try:
        experiments = []
        fitted_by_key = {}
        for key in benchmark_keys:
            say(f"Fitting {models.DISPLAY_NAME[key]}")
            hp = params.xgb_params() if key == PRODUCTION_KEY else models.default_params(key, params.seed)
            fitted = models.fit_model(key, hp, X_tr, y_tr, X_val, y_val)
            fitted_by_key[key] = fitted
            val_pred = fitted.predict(X_val)
            status = "active" if key == PRODUCTION_KEY else ("reference" if key == "baseline" else "benchmark")
            experiments.append({
                "key": key, "name": models.DISPLAY_NAME[key], "family": models.FAMILY[key],
                "status": status, "hyperparameters": fitted.hyperparameters,
                "validation": models.regression_metrics(y_val, val_pred),
                "fit_seconds": fitted.fit_seconds, "train_rows": int(len(X_tr)),
                **({"best_iteration": fitted.best_iteration} if fitted.best_iteration is not None else {}),
            })

        prod = fitted_by_key[PRODUCTION_KEY]
        prod.model.save_model(vdir / "model.ubj")
        val_pred = prod.predict(X_val)

        say("Calibrating prediction band on the calibration split")
        band = fit_band(X_cal, y_cal, prod.predict(X_cal))
        band_val = band_coverage(band, X_val, y_val, val_pred)

        say("Computing diagnostics on the validation split")
        telemetry = training_telemetry(train_df, X_val, y_val, val_pred)
        diagnostics = {
            "curves": {
                "metric": "RMSE of log1p(duration) (= RMSLE)",
                "train": downsample_curve(prod.curves["train_rmse"]),
                "validation": downsample_curve(prod.curves["validation_rmse"]),
                "train_curve_rows": prod.curves["train_curve_rows"],
                "best_iteration": prod.best_iteration,
                "rounds_run": len(prod.curves["validation_rmse"]),
            },
            **telemetry,
            "deep_dive": deep_dive(X_val, y_val, val_pred),
        }
        prod_exp = next(e for e in experiments if e["key"] == PRODUCTION_KEY)
        metadata = {
            "model_name": "NYC trip-duration regressor",
            "active_model": models.DISPLAY_NAME[PRODUCTION_KEY],
            "active_model_key": PRODUCTION_KEY,
            "version": version,
            "trained_at": utc_now_iso(),
            "training_seconds": None,  # filled below
            "train_params": asdict(params),
            "hyperparameters": prod.hyperparameters,
            "early_stopping_rounds": config.EARLY_STOPPING_ROUNDS,
            "best_iteration": prod.best_iteration,
            "features": list(BASE_FEATURES),
            "feature_count": len(BASE_FEATURES),
            "feature_importance": models.gain_importance(prod, list(BASE_FEATURES)),
            "target": "log1p(trip_duration seconds)",
            "rows": {"train": int(len(X_tr)), "calibration": int(len(X_cal)), "validation": int(len(X_val))},
            "data_report": trips.report(),
            "training_pickup_range": {"min": train_df["pickup_datetime"].min().isoformat(),
                                      "max": train_df["pickup_datetime"].max().isoformat()},
            "validation_metrics": prod_exp["validation"],
            "band": band,
            "band_validation_coverage": band_val,
        }
        write_json_atomic(vdir / "experiments.json", experiments)
        write_json_atomic(vdir / "diagnostics.json", diagnostics)
        metadata["training_seconds"] = time.perf_counter() - t_start
        write_json_atomic(vdir / "metadata.json", metadata)

        say("Validating artifacts before promotion")
        store.validate_version(version)
        store.promote(version)
        say(f"Promoted {version}")
        return version
    except BaseException:
        store.discard(version)
        raise
