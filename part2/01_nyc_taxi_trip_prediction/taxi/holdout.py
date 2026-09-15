"""One-time final holdout evaluation on the test split.

This is the only module that reads the test role. It is invoked explicitly via
`scripts/finalize.py`, never by retraining, AutoResearch, or the UI.
"""
from __future__ import annotations

from . import config, models
from .artifacts import ArtifactStore, utc_now_iso, write_json_atomic
from .data import TripStore
from .diagnostics import band_coverage, deep_dive
from .features import build_features


class HoldoutAlreadyEvaluated(RuntimeError):
    pass


def finalize(version: str | None = None, force: bool = False, trips: TripStore | None = None,
             store: ArtifactStore | None = None) -> dict:
    trips = trips or TripStore()
    store = store or ArtifactStore()
    version = version or store.current_version()
    if version is None:
        raise RuntimeError("No active model version to finalize. Train first.")
    bundle = store.validate_version(version)
    existing = bundle.path / "holdout.json"
    if existing.exists() and not force:
        raise HoldoutAlreadyEvaluated(
            f"Version {version} already has a final holdout evaluation. "
            "Re-running on the test split is refused unless --force is given (and is logged)."
        )

    test_df = trips.role("test")
    X_te = build_features(test_df)
    y_te = models.to_log_target(test_df[config.TARGET_COLUMN])
    pred = bundle.predict_log(X_te)
    result = {
        "version": version,
        "evaluated_at": utc_now_iso(),
        "population": "test split (final holdout)",
        "metrics": models.regression_metrics(y_te, pred),
        "band_coverage": band_coverage(bundle.metadata["band"], X_te, y_te, pred),
        "deep_dive_tiers": deep_dive(X_te, y_te, pred)["tiers"],
        "forced_rerun": bool(existing.exists()),
    }
    write_json_atomic(existing, result)
    log = store.read_holdout_log()
    log.append({"version": version, "evaluated_at": result["evaluated_at"],
                "rmsle": result["metrics"]["rmsle"], "forced_rerun": result["forced_rerun"]})
    write_json_atomic(store.holdout_log, log)
    return result
