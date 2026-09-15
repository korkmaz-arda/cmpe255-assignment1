"""Prediction band, charting telemetry and deep-dive diagnostics — all computed, none hard-coded."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .features import BASE_FEATURES, TIER_LABELS, TIER_ORDER, distance_tier
from .models import from_log_target, regression_metrics


# --- Prediction band (F06) ---------------------------------------------------
def fit_band(features: pd.DataFrame, y_log: np.ndarray, pred_log: np.ndarray,
             level: float = config.BAND_LEVEL) -> dict:
    """Empirical residual quantiles in log space, per distance tier, fitted on the calibration split."""
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    resid = np.asarray(y_log) - np.asarray(pred_log)
    tiers = distance_tier(features)
    out = {"level": level, "quantiles": [lo_q, hi_q],
           "overall": {"low": float(np.quantile(resid, lo_q)), "high": float(np.quantile(resid, hi_q)),
                       "n": int(len(resid))},
           "tiers": {}}
    for t in TIER_ORDER:
        r = resid[tiers == t]
        if len(r) >= config.BAND_MIN_TIER_ROWS:
            out["tiers"][t] = {"low": float(np.quantile(r, lo_q)), "high": float(np.quantile(r, hi_q)),
                               "n": int(len(r))}
    return out


def band_offsets(band: dict, tiers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low = np.array([band["tiers"].get(t, band["overall"])["low"] for t in tiers], dtype=float)
    high = np.array([band["tiers"].get(t, band["overall"])["high"] for t in tiers], dtype=float)
    return low, high


def band_coverage(band: dict, features: pd.DataFrame, y_log: np.ndarray, pred_log: np.ndarray) -> dict:
    tiers = distance_tier(features)
    low, high = band_offsets(band, tiers)
    inside = (y_log >= pred_log + low) & (y_log <= pred_log + high)
    by_tier = {t: {"coverage": float(inside[tiers == t].mean()), "n": int((tiers == t).sum())}
               for t in TIER_ORDER if (tiers == t).any()}
    return {"coverage": float(inside.mean()), "n": int(len(inside)), "by_tier": by_tier,
            "nominal": band["level"]}


# --- Charting telemetry (F05) ------------------------------------------------
def downsample_curve(values: list[float], points: int = 30) -> list[dict]:
    n = len(values)
    if n == 0:
        return []
    idx = sorted(set(np.linspace(0, n - 1, num=min(points, n)).round().astype(int).tolist()))
    return [{"iteration": int(i + 1), "value": float(values[i])} for i in idx]


def fixed_histogram(values: np.ndarray, lo: float, hi: float, bins: int) -> dict:
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts],
            "below_range": int((values < lo).sum()), "above_range": int((values > hi).sum())}


def training_telemetry(train_df: pd.DataFrame, val_features: pd.DataFrame, y_val_log: np.ndarray,
                       val_pred_log: np.ndarray, seed: int = 0) -> dict:
    """Validation residual sample/histogram and training-sample distributions."""
    actual_min = from_log_target(y_val_log) / 60.0
    pred_min = from_log_target(val_pred_log) / 60.0
    err_min = pred_min - actual_min
    rng = np.random.default_rng(seed)
    k = min(400, len(actual_min))
    idx = np.sort(rng.choice(len(actual_min), size=k, replace=False))
    dur_min = train_df["trip_duration"].to_numpy() / 60.0
    ts = pd.to_datetime(train_df["pickup_datetime"])
    weekday = ts.dt.dayofweek.to_numpy() < 5
    hours = ts.dt.hour.to_numpy()
    return {
        "residual_sample": {
            "population": "validation split",
            "rows": [{"actual_min": float(actual_min[i]), "predicted_min": float(pred_min[i]),
                      "error_min": float(err_min[i])} for i in idx],
        },
        "residual_histogram": {"population": "validation split", "unit": "minutes (predicted − actual)",
                               **fixed_histogram(err_min, -15.0, 15.0, 15)},
        "duration_histogram": {"population": "training sample", "unit": "minutes",
                               **fixed_histogram(dur_min, 1.0, 60.0, 12)},
        "hourly_pickups": {
            "population": "training sample",
            "weekday": np.bincount(hours[weekday], minlength=24).astype(int).tolist(),
            "weekend": np.bincount(hours[~weekday], minlength=24).astype(int).tolist(),
        },
    }


# --- Deep-dive (S04) ---------------------------------------------------------
def _mape(sec_true, sec_pred) -> float:
    return float(np.mean(np.abs(sec_pred - sec_true) / sec_true) * 100.0)


def deep_dive(val_features: pd.DataFrame, y_val_log: np.ndarray, val_pred_log: np.ndarray) -> dict:
    tiers = distance_tier(val_features)
    sec_true = from_log_target(y_val_log)
    sec_pred = from_log_target(val_pred_log)
    tier_rows = []
    for t in TIER_ORDER:
        m = tiers == t
        if m.sum() < 2:
            continue
        met = regression_metrics(y_val_log[m], val_pred_log[m])
        tier_rows.append({"tier": t, "label": TIER_LABELS[t], "mae_s": met["mae_s"],
                          "mape_pct": _mape(sec_true[m], sec_pred[m]), "r2": met["r2"],
                          "rmsle": met["rmsle"], "n": int(m.sum())})
    resid_log = val_pred_log - y_val_log
    residuals = {
        "space": "log1p residual (predicted − actual)",
        "skewness": float(stats.skew(resid_log)),
        "excess_kurtosis": float(stats.kurtosis(resid_log)),
        "mean_bias_log": float(resid_log.mean()),
        "mean_bias_error_s": float(np.mean(sec_pred - sec_true)),
        "mape_pct": _mape(sec_true, sec_pred),
        "ideal": {"skewness": 0.0, "excess_kurtosis": 0.0, "mean_bias_error_s": 0.0, "mape_pct": 0.0},
    }
    correlations = []
    for f in BASE_FEATURES:
        x = val_features[f].to_numpy()
        if np.std(x) == 0:
            correlations.append({"feature": f, "pearson_r": None, "p_value": None, "signal": "constant"})
            continue
        r, p = stats.pearsonr(x, y_val_log)
        a = abs(r)
        signal = "strong" if a >= 0.5 else "moderate" if a >= 0.3 else "weak" if a >= 0.1 else "negligible"
        correlations.append({"feature": f, "pearson_r": float(r), "p_value": float(p), "signal": signal})
    correlations.sort(key=lambda c: -abs(c["pearson_r"] or 0.0))
    return {"population": "validation split", "n": int(len(y_val_log)), "tiers": tier_rows,
            "residuals": residuals, "correlations": correlations}
