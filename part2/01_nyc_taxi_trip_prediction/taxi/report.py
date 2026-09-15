"""CRISP-DM report content generated from the live artifacts (S05). Framework-independent.

Each phase is a list of blocks: ("md", text) | ("latex", expr) | ("table", list[dict]).
"""
from __future__ import annotations

from . import config
from .features import BASE_FEATURES, FEATURE_DESCRIPTIONS

PHASES = [
    ("business", "Business Understanding"),
    ("data", "Data Understanding"),
    ("preparation", "Data Preparation & Feature Engineering"),
    ("modeling", "Modeling & Benchmarks"),
    ("evaluation", "Evaluation & Diagnostics"),
    ("deployment", "Deployment & AutoResearch"),
]


def _f(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def build_report(metadata: dict | None, experiments: list | None, diagnostics: dict | None,
                 holdout: dict | None, session: dict | None, data_report: dict | None) -> dict[str, list]:
    md = metadata or {}
    dr = data_report or md.get("data_report") or {}
    vm = md.get("validation_metrics") or {}
    out: dict[str, list] = {}

    out["business"] = [
        ("md", "**Objective.** Estimate how long a New York City yellow-cab ride will take *before* it starts, "
               "from what is known at request time: pickup and dropoff location, pickup time, and passenger count. "
               "The estimate also drives an illustrative fare quote, so duration errors carry through to price."),
        ("md", "**Success criterion.** Lower RMSLE (root mean squared logarithmic error) than simple references "
               "on held-out trips. RMSLE penalises *relative* error, so being 2 minutes off on a 5-minute trip "
               "counts more than 2 minutes off on a 60-minute trip:"),
        ("latex", r"\mathrm{RMSLE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}\big(\log(1+\hat y_i)-\log(1+y_i)\big)^2}"),
        ("md", "**Scope and limits.** The model learns traffic patterns from January–June 2016. It says nothing "
               "about present-day traffic, road closures or weather, and a prediction for another period is "
               "flagged as temporal extrapolation."),
    ]

    rows = [{"Rule": c["description"], "Rows before": f"{c['rows_before']:,}", "Removed": f"{c['removed']:,}"}
            for c in dr.get("cleaning", [])]
    out["data"] = [
        ("md", f"**Source.** {dr.get('source', 'Kaggle NYC Taxi Trip Duration train.csv')}. Loaded from the local "
               "archive; no network access or credentials are needed. The Kaggle `test.csv` has no duration label, "
               "so it is not used for evaluation."),
        ("md", f"**Size.** {dr.get('raw_rows', 0):,} raw trips; pickups from {dr.get('pickup_datetime_min', '?')} "
               f"to {dr.get('pickup_datetime_max', '?')}."),
        ("table", [
            {"Column": "id", "Meaning": "Trip identifier", "Use": "Stable split assignment only (never a feature)"},
            {"Column": "vendor_id", "Meaning": "Technology provider (1 or 2)", "Use": "Not used"},
            {"Column": "pickup_datetime", "Meaning": "Meter engaged", "Use": "Calendar features"},
            {"Column": "dropoff_datetime", "Meaning": "Meter disengaged", "Use": "Not a feature (post-trip)"},
            {"Column": "passenger_count", "Meaning": "Driver-entered passengers", "Use": "Feature; valid 1–6"},
            {"Column": "pickup/dropoff lat/lon", "Meaning": "GPS coordinates", "Use": "Spatial features; NYC box"},
            {"Column": "store_and_fwd_flag", "Meaning": "Record held in vehicle memory", "Use": "Not used"},
            {"Column": "trip_duration", "Meaning": "Seconds (target)", "Use": "Target; valid 60 s – 3 h"},
        ]),
        ("md", "The Kaggle file has no missing values in these columns; problems show up as out-of-range values "
               "instead (zero coordinates, multi-day durations, zero passengers)."),
    ]

    split = dr.get("split_sizes", {})
    out["preparation"] = [
        ("md", "**Cleaning.** Only domain rules are applied, in this order (counts are from this dataset):"),
        ("table", rows),
        ("md", f"**{dr.get('clean_rows', 0):,} usable trips.** Zero-displacement trips are kept: they are valid "
               "records, and dropping them would make the scores look better than they are."),
        ("md", "**Splits by stable hash of trip id** (independent of row order, sample size and seed): "
               f"fit pool {split.get('fit', 0):,} · calibration {split.get('calibration', 0):,} · "
               f"validation {split.get('validation', 0):,} · test {split.get('test', 0):,}. "
               "The fit pool trains models; calibration sets the prediction band only; validation drives early "
               "stopping, comparisons, AutoResearch and diagnostics; test is scored once in a final holdout "
               "evaluation."),
        ("md", "**Target transform.** Durations are right-skewed, so models learn the log of duration and "
               "predictions are converted back to seconds:"),
        ("latex", r"y' = \log(1+\text{duration}_s), \qquad \hat{\text{duration}}_s = e^{\hat y'} - 1"),
        ("md", "**City-block distance** is the east–west leg (along the pickup latitude) plus the north–south "
               "leg, each measured as great-circle kilometres, not raw degree differences:"),
        ("latex", r"d_{\text{block}} = \mathrm{hav}(\varphi_1,\lambda_1;\varphi_1,\lambda_2) + "
                  r"\mathrm{hav}(\varphi_1,\lambda_1;\varphi_2,\lambda_1)"),
        ("table", [{"Feature": f, "Definition": FEATURE_DESCRIPTIONS[f]} for f in BASE_FEATURES]),
        ("md", "**Leakage guard.** Features are built only from request-time columns; dropoff time, duration "
               "and id cannot reach the model."),
    ]

    exp_rows = [{"Model": e["name"], "Role": e["status"], "Validation RMSLE": _f(e["validation"]["rmsle"]),
                 "Validation R²": _f(e["validation"]["r2"], 3), "Fit time (s)": _f(e["fit_seconds"], 1)}
                for e in (experiments or [])]
    out["modeling"] = [
        ("md", f"Every model below was fitted on the same {md.get('rows', {}).get('train', 0):,} training trips "
               "and scored on the same validation split."),
        ("table", exp_rows),
        ("md", f"**Active model:** {md.get('active_model', 'n/a')} with "
               f"`{md.get('hyperparameters', {})}`; early stopping after {md.get('early_stopping_rounds', '?')} "
               f"rounds without validation improvement (best iteration {md.get('best_iteration', '?')})."),
        ("md", "The median baseline is a reference computed here, not an external benchmark. Kaggle leaderboard "
               "scores use a different test set and protocol, so they are not comparable and are not shown."),
    ]

    cov = md.get("band_validation_coverage") or {}
    dd = (diagnostics or {}).get("deep_dive") or {}
    res = dd.get("residuals") or {}
    evaluation = [
        ("md", f"**Validation (used for model selection):** RMSLE {_f(vm.get('rmsle'))}, R² {_f(vm.get('r2'), 3)}, "
               f"MAE {_f(vm.get('mae_s'), 1)} s over {vm.get('n', 0):,} trips. Because these rows guide "
               "selection, the numbers can be slightly optimistic."),
        ("md", f"**Prediction band:** a {int(md.get('band', {}).get('level', 0.9) * 100)}% band from residual "
               "quantiles on the calibration split, computed separately for each distance tier. Descriptive "
               f"coverage on validation: {_f(cov.get('coverage'), 3)}."),
        ("md", f"**Residual shape (validation, log space):** skewness {_f(res.get('skewness'), 3)}, excess "
               f"kurtosis {_f(res.get('excess_kurtosis'), 3)}, mean bias {_f(res.get('mean_bias_error_s'), 1)} s, "
               f"MAPE {_f(res.get('mape_pct'), 1)}%."),
    ]
    if holdout:
        hm = holdout["metrics"]
        evaluation.append(("md", f"**Final holdout evaluation (test, run once on {holdout['evaluated_at']}):** "
                                 f"RMSLE {_f(hm['rmsle'])}, R² {_f(hm['r2'], 3)}, MAE {_f(hm['mae_s'], 1)} s over "
                                 f"{hm['n']:,} trips; band coverage {_f(holdout['band_coverage']['coverage'], 3)}."))
    else:
        evaluation.append(("md", "**Final holdout evaluation:** this version has not been finalized, so no test "
                                 "score exists yet. Run `python scripts/finalize.py` once, after the configuration "
                                 "is settled."))
    evaluation.append(("md", "**Split design limitation:** splits are random over Jan–Jun 2016 (like the Kaggle "
                             "setup), not time-ordered, so the scores do not measure performance on future months."))
    out["evaluation"] = evaluation

    deploy = [
        ("md", "**Serving.** A single local Streamlit app loads the active model version named in "
               "`artifacts/current.json`. Features for each request are built by the same function used in "
               "training. There is no separate API, no background worker, and no service-level guarantee."),
        ("md", "**Retraining.** The retrain dialog runs the full pipeline synchronously: sample, fit four models, "
               "calibrate, compute diagnostics, write a new immutable version, check that it loads and predicts, "
               "then atomically switch the pointer to it. If anything fails, the previous version stays live. New "
               "versions start *not finalized*."),
    ]
    if session:
        s = session["summary"]
        deploy.append(("md", f"**Latest AutoResearch session** ({session['session_id']}): XGBoost configuration "
                             f"improved from {_f(s['initial_rmsle'], 5)} to {_f(s['best_rmsle'], 5)} validation "
                             f"RMSLE ({s['accepted']} accepted / {s['rejected']} rejected). The search never reads "
                             "the test split and does not replace the served model."))
    else:
        deploy.append(("md", "No AutoResearch session has been run yet."))
    out["deployment"] = deploy
    return out
