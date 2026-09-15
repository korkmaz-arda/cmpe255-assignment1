"""AutoResearch: a greedy, validation-only hill-climbing model search (F12/F14).

Phases
  1. backbone tournament  - six families on the base features (leaderboard + champion)
  2. feature mutation     - candidate derived features added to the XGBoost configuration
  3. hyperparameter annealing - named XGBoost configurations
  4. ensemble blending    - evolved XGBoost blended with HistGradientBoosting

Like-for-like gate: phases 2-4 compare every candidate with the *current incumbent of the same
configuration being mutated* (initially the XGBoost base configuration's own validation RMSLE).
A candidate is accepted only if it improves RMSLE by more than epsilon; otherwise it is reverted.

Evaluation population: always the full fixed validation split. The test split is never read.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from . import config, models
from .artifacts import ArtifactStore, utc_now_iso
from .data import TripStore, sample_rows
from .features import BASE_FEATURES, build_features

# Each candidate's `expr` is the exact code evaluated to create the column (shown verbatim in the UI).
FEATURE_CANDIDATES = [
    {"name": "log_manhattan_km", "expr": "np.log1p(df['distance_manhattan_km'])",
     "hypothesis": "Duration grows sub-linearly with distance (fixed boarding/stoplight overhead), "
                   "so a log-distance feature may be easier to split on."},
    {"name": "tortuosity", "expr": "df['distance_manhattan_km'] / (df['distance_haversine_km'] + 1e-3)",
     "hypothesis": "How much longer the city-block path is than the straight line may capture street-grid detours."},
    {"name": "airport_expressway",
     "expr": "((np.minimum(df['dist_jfk_km'], df['dist_lga_km']) <= 2.0) & (df['distance_haversine_km'] > 8.0)).astype('float64')",
     "hypothesis": "Long trips touching JFK/LGA use expressways and may move faster than distance alone implies."},
    {"name": "hour_sin", "expr": "np.sin(2 * np.pi * df['hour'] / 24.0)",
     "hypothesis": "A cyclical encoding makes 23:00 and 00:00 neighbours for the model."},
    {"name": "hour_cos", "expr": "np.cos(2 * np.pi * df['hour'] / 24.0)",
     "hypothesis": "The cosine half of the cyclical hour encoding disambiguates the sine component."},
    {"name": "rush_x_distance", "expr": "df['is_rush_hour'] * df['distance_manhattan_km']",
     "hypothesis": "Weekday congestion should cost more time per kilometre, an interaction trees may find faster if given explicitly."},
    {"name": "displacement_efficiency",
     "expr": "df['distance_haversine_km'] / (df['distance_manhattan_km'] + 1e-3)",
     "hypothesis": "Straight-line share of the city-block path, the inverse view of tortuosity."},
    {"name": "late_night_long_trip",
     "expr": "((df['is_late_night'] > 0) & (df['distance_haversine_km'] > 5.0)).astype('float64')",
     "hypothesis": "Long late-night trips face empty roads and may be disproportionately fast."},
]

HYPERPARAMETER_CANDIDATES = [
    {"name": "deeper_trees", "override": {"max_depth": 10, "min_child_weight": 10},
     "hypothesis": "Deeper trees can model higher-order spatial x temporal interactions."},
    {"name": "higher_subsampling", "override": {"subsample": 0.95, "colsample_bytree": 0.95},
     "hypothesis": "With abundant data, less row/column subsampling may reduce bias more than it adds variance."},
    {"name": "l2_regularization", "override": {"reg_lambda": 10.0},
     "hypothesis": "Stronger L2 on leaf weights may curb overfitting to noisy GPS/clock records."},
    {"name": "more_rounds_lower_lr", "override": {"n_estimators": 600, "learning_rate": 0.04},
     "hypothesis": "Smaller steps over more rounds usually generalise better under early stopping."},
    {"name": "shallow_fast", "override": {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.15},
     "hypothesis": "A shallower, faster learner may match accuracy at lower cost if interactions are simple."},
]

BLEND_WEIGHTS = [0.5, 0.7]  # weight on the evolved XGBoost model; remainder on HistGradientBoosting
BACKBONE_KEYS = ["xgboost", "hist_gb", "extra_trees", "random_forest", "mlp", "ridge"]
BOOSTING_KEYS = {"xgboost", "hist_gb"}


def add_feature(df: pd.DataFrame, candidate: dict) -> pd.DataFrame:
    out = df.copy()
    out[candidate["name"]] = eval(candidate["expr"], {"np": np, "__builtins__": {}}, {"df": df})  # noqa: S307
    return out


def gate(candidate_rmsle: float, incumbent_rmsle: float, epsilon: float = config.AUTORESEARCH_EPSILON) -> bool:
    return (incumbent_rmsle - candidate_rmsle) > epsilon


def _reflection(decision: str, component: str, cand: float, inc: float, eps: float, what: str) -> str:
    delta = inc - cand
    if decision == "ACCEPTED":
        return (f"{component} scored {cand:.5f} against incumbent {inc:.5f} "
                f"(improvement {delta:+.5f} > ε={eps:g}); {what} is kept and becomes the new incumbent.")
    return (f"{component} scored {cand:.5f} against incumbent {inc:.5f} "
            f"(improvement {delta:+.5f} ≤ ε={eps:g}); {what} is reverted and the incumbent is unchanged.")


def run_session(fit_sample_size: int = 150_000, seed: int = 42, trips: TripStore | None = None,
                store: ArtifactStore | None = None, progress: Callable[[str], None] | None = None,
                epsilon: float = config.AUTORESEARCH_EPSILON, backbone_keys=BACKBONE_KEYS,
                feature_candidates=FEATURE_CANDIDATES, hyperparameter_candidates=HYPERPARAMETER_CANDIDATES,
                blend_weights=BLEND_WEIGHTS, fit_fn=models.fit_model, save: bool = True) -> dict:
    trips = trips or TripStore()
    store = store or ArtifactStore()
    say = progress or (lambda m: None)
    started = utc_now_iso()
    t0 = time.perf_counter()

    say("Loading fit sample and fixed validation split")
    train_df = sample_rows(trips.role("fit"), fit_sample_size, seed)
    val_df = trips.role("validation")
    X_tr_base, y_tr = build_features(train_df), models.to_log_target(train_df[config.TARGET_COLUMN])
    X_val_base, y_val = build_features(val_df), models.to_log_target(val_df[config.TARGET_COLUMN])

    steps: list[dict] = []

    def record(**kw):
        kw.setdefault("timestamp", utc_now_iso())
        kw["iteration"] = len(steps) + 1
        steps.append(kw)
        return kw

    # -- Phase 1: backbone tournament -------------------------------------------------------
    leaderboard = []
    base_preds: dict[str, np.ndarray] = {}
    for key in backbone_keys:
        say(f"Backbone tournament: {models.DISPLAY_NAME[key]}")
        params = models.default_params(key, seed)
        fitted = fit_fn(key, params, X_tr_base, y_tr, X_val_base, y_val)
        t_inf = time.perf_counter()
        pred = fitted.predict(X_val_base)
        latency_us = (time.perf_counter() - t_inf) / len(X_val_base) * 1e6
        base_preds[key] = pred
        met = models.regression_metrics(y_val, pred)
        leaderboard.append({"key": key, "name": models.DISPLAY_NAME[key], "family": models.FAMILY[key],
                            **met, "fit_seconds": fitted.fit_seconds, "latency_us_per_row": latency_us,
                            "hyperparameters": params})
        record(phase="backbone", category="model family", component=models.DISPLAY_NAME[key],
               hypothesis=f"Benchmark {models.FAMILY[key]} on the 20 base features.",
               code=f"model = fit_model({key!r}, {params!r})", params=params,
               rmsle_candidate=met["rmsle"], incumbent_before=None, incumbent_after=None, delta=None,
               decision="BENCHMARK",
               reflection=f"{models.DISPLAY_NAME[key]} reached validation RMSLE {met['rmsle']:.5f} "
                          f"(R² {met['r2']:.3f}) in {fitted.fit_seconds:.1f} s.")
    leaderboard.sort(key=lambda r: r["rmsle"])
    for rank, row in enumerate(leaderboard, start=1):
        row["rank"] = rank
        row["champion"] = rank == 1

    # -- Hill-climb state: the XGBoost configuration being mutated -----------------------------
    xgb_params = models.default_params("xgboost", seed)
    active_features = list(BASE_FEATURES)
    X_tr, X_val = X_tr_base, X_val_base
    incumbent = float(models.regression_metrics(y_val, base_preds["xgboost"])["rmsle"]) \
        if "xgboost" in base_preds else None
    if incumbent is None:
        fitted = fit_fn("xgboost", xgb_params, X_tr, y_tr, X_val, y_val)
        base_preds["xgboost"] = fitted.predict(X_val)
        incumbent = models.regression_metrics(y_val, base_preds["xgboost"])["rmsle"]
    initial_rmsle = incumbent
    incumbent_pred = base_preds["xgboost"]
    incumbent_kind = "xgboost"

    def gated_step(phase, category, component, hypothesis, code, params, cand_rmsle, what):
        nonlocal incumbent
        before = incumbent
        accepted = gate(cand_rmsle, before, epsilon)
        decision = "ACCEPTED" if accepted else "REJECTED"
        if accepted:
            incumbent = cand_rmsle
        record(phase=phase, category=category, component=component, hypothesis=hypothesis, code=code,
               params=params, features=list(active_features), rmsle_candidate=cand_rmsle,
               incumbent_before=before, incumbent_after=incumbent, delta=before - cand_rmsle,
               decision=decision, reflection=_reflection(decision, component, cand_rmsle, before, epsilon, what))
        return accepted

    # -- Phase 2: feature mutation ----------------------------------------------------------
    for cand in feature_candidates:
        say(f"Feature mutation: {cand['name']}")
        X_tr_c, X_val_c = add_feature(X_tr, cand), add_feature(X_val, cand)
        fitted = fit_fn("xgboost", xgb_params, X_tr_c, y_tr, X_val_c, y_val)
        pred = fitted.predict(X_val_c)
        score = models.regression_metrics(y_val, pred)["rmsle"]
        trial_features = active_features + [cand["name"]]
        before_features = active_features
        active_features = trial_features  # recorded as the tried feature set
        accepted = gated_step("feature", "feature engineering", cand["name"], cand["hypothesis"],
                              f"df[{cand['name']!r}] = {cand['expr']}", dict(xgb_params), score,
                              f"feature `{cand['name']}`")
        if accepted:
            X_tr, X_val = X_tr_c, X_val_c
            incumbent_pred = pred
        else:
            active_features = before_features

    # -- Phase 3: hyperparameter annealing --------------------------------------------------
    for cand in hyperparameter_candidates:
        say(f"Hyperparameter annealing: {cand['name']}")
        trial = {**xgb_params, **cand["override"]}
        fitted = fit_fn("xgboost", trial, X_tr, y_tr, X_val, y_val)
        pred = fitted.predict(X_val)
        score = models.regression_metrics(y_val, pred)["rmsle"]
        accepted = gated_step("hyperparameter", "XGBoost configuration", cand["name"], cand["hypothesis"],
                              f"params.update({cand['override']!r})", trial, score,
                              f"configuration `{cand['name']}`")
        if accepted:
            xgb_params = trial
            incumbent_pred = pred

    # -- Phase 4: ensemble blending ---------------------------------------------------------
    blend_partner = None
    if blend_weights:
        boosting = [r for r in leaderboard if r["key"] in BOOSTING_KEYS]
        partner_key = "hist_gb" if any(r["key"] == "hist_gb" for r in boosting) else None
        if partner_key:
            say("Ensemble blending: fitting HistGradientBoosting on the current feature set")
            partner = fit_fn(partner_key, models.default_params(partner_key, seed), X_tr, y_tr, X_val, y_val)
            partner_pred = partner.predict(X_val)
            xgb_pred = incumbent_pred  # evolved XGBoost predictions (phases 2-3)
            blend_partner = partner_key
            for w in blend_weights:
                say(f"Ensemble blending: {w:.0%} XGBoost / {1 - w:.0%} HistGB")
                pred = w * xgb_pred + (1 - w) * partner_pred
                score = models.regression_metrics(y_val, pred)["rmsle"]
                name = f"blend_xgb{int(round(w * 100))}_hgb{int(round((1 - w) * 100))}"
                accepted = gated_step("ensemble", "model blending", name,
                                      f"Averaging the evolved XGBoost with HistGradientBoosting in log space "
                                      f"(weights {w:.1f}/{1 - w:.1f}) may cancel uncorrelated errors.",
                                      f"pred = {w:.1f} * xgb_pred + {1 - w:.1f} * hist_gb_pred  # log1p space",
                                      {"xgboost": dict(xgb_params), "hist_gb": models.default_params(partner_key, seed),
                                       "weights": [w, 1 - w]}, score, f"blend `{name}`")
                if accepted:
                    incumbent_pred = pred
                    incumbent_kind = name

    gated = [s for s in steps if s["decision"] in ("ACCEPTED", "REJECTED")]
    session = {
        "session_id": datetime.now(timezone.utc).strftime("ar-%Y%m%dT%H%M%SZ"),
        "started_at": started,
        "finished_at": utc_now_iso(),
        "runtime_seconds": time.perf_counter() - t0,
        "config": {"fit_sample_size": int(len(X_tr_base)), "seed": seed, "epsilon": epsilon,
                   "evaluation_population": "full fixed validation split",
                   "validation_rows": int(len(X_val_base)), "split_salt": config.SPLIT_SALT,
                   "test_split_used": False},
        "summary": {
            "initial_rmsle": initial_rmsle,
            "initial_configuration": "XGBoost base configuration on the 20 base features",
            "best_rmsle": incumbent,
            "improvement_pct": (initial_rmsle - incumbent) / initial_rmsle * 100.0,
            "total_steps": len(steps),
            "gated_steps": len(gated),
            "accepted": sum(s["decision"] == "ACCEPTED" for s in gated),
            "rejected": sum(s["decision"] == "REJECTED" for s in gated),
            "champion_backbone": leaderboard[0]["name"] if leaderboard else None,
            "final_model": incumbent_kind,
            "blend_partner": blend_partner,
            "final_xgboost_params": xgb_params,
        },
        "leaderboard": leaderboard,
        "active_features": active_features,
        "steps": steps,
    }
    if save:
        store.save_session(session)
    say(f"Session finished: best validation RMSLE {incumbent:.5f}")
    return session
