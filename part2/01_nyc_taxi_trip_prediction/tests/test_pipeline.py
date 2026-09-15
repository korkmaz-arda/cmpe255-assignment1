import json

import numpy as np
import pandas as pd
import pytest

from taxi import config, holdout, landmarks
from taxi.artifacts import ArtifactStore
from taxi.data import TripStore
from taxi.predict import fare_components, predict_batch, predict_frame, predict_trip
from taxi.features import build_features
from taxi.train import TrainParams, run_training

from conftest import FAST_PARAMS


@pytest.fixture(scope="module")
def trained(tmp_path_factory, processed_path):
    root = tmp_path_factory.mktemp("artifacts")
    trips = TripStore(processed_path)
    store = ArtifactStore(root)
    version = run_training(TrainParams(seed=1, **FAST_PARAMS), trips=trips, store=store)
    return {"trips": trips, "store": store, "version": version, "root": root}


def request(**over):
    ts, wall = landmarks.get("times_square"), landmarks.get("wall_street")
    kw = dict(pickup_latitude=ts.lat, pickup_longitude=ts.lon, dropoff_latitude=wall.lat,
              dropoff_longitude=wall.lon, pickup_datetime=pd.Timestamp("2016-03-15 08:30"), passenger_count=1)
    kw.update(over)
    return kw


def test_training_never_reads_test_split(trained):
    assert "test" not in trained["trips"].accessed
    assert {"fit", "calibration", "validation"} <= set(trained["trips"].accessed)


def test_round_trip_promote_reload_predict(trained):
    store = trained["store"]
    assert store.current_version() == trained["version"]
    bundle = store.load_current()
    md = bundle.metadata
    assert md["feature_count"] == 20 and md["features"] == list(build_features(pd.DataFrame([{
        "pickup_datetime": pd.Timestamp("2016-01-01"), "passenger_count": 1, "pickup_latitude": 40.7,
        "pickup_longitude": -74.0, "dropoff_latitude": 40.71, "dropoff_longitude": -74.0}])).columns)
    assert {e["status"] for e in bundle.experiments} == {"reference", "benchmark", "active"}
    assert "validation" in bundle.experiments[0] and "test" not in json.dumps(bundle.experiments)
    assert bundle.holdout is None
    out = predict_trip(bundle, **request())
    assert out["duration"]["seconds"] >= config.MIN_PREDICTED_DURATION_S
    assert out["band"]["low_seconds"] <= out["duration"]["seconds"] <= out["band"]["high_seconds"]
    assert out["fare"]["total"] == pytest.approx(sum(i["amount"] for i in out["fare"]["items"]), abs=0.011)


def test_calibration_rows_never_used_for_fitting(trained):
    md = trained["store"].load_current().metadata
    trips = trained["trips"]
    fit_ids = set(trips.role("fit")["id"])
    cal_ids = set(trips.role("calibration")["id"])
    assert not fit_ids & cal_ids
    assert md["rows"]["train"] == len(fit_ids)  # full fit pool, nothing else
    assert md["band"]["overall"]["n"] == len(cal_ids)


def test_predictions_are_deterministic_and_leakage_free(trained):
    bundle = trained["store"].load_current()
    base = pd.DataFrame([{k: v for k, v in request().items()}])
    a = base.assign(id="a", dropoff_datetime=pd.Timestamp("2016-03-15 08:35"), trip_duration=300)
    b = base.assign(id="b", dropoff_datetime=pd.Timestamp("2016-03-15 12:00"), trip_duration=12000)
    pa, pb = predict_frame(bundle, a), predict_frame(bundle, b)
    pd.testing.assert_frame_equal(pa, pb)
    assert predict_trip(bundle, **request()) == predict_trip(bundle, **request())


@pytest.mark.parametrize("field", ["pickup_datetime", "passenger_count"])
def test_core_rejects_missing_inputs(trained, field):
    bundle = trained["store"].load_current()
    with pytest.raises(ValueError, match=field):
        predict_trip(bundle, **request(**{field: None}))


def test_batch_reports_missing_rows_and_counts(trained):
    bundle = trained["store"].load_current()
    rows = [request(), request(passenger_count=3)]
    res = predict_batch(bundle, rows)
    assert res["count"] == 2 and len(res["predictions"]) == 2
    with pytest.raises(ValueError, match="row"):
        predict_batch(bundle, [request(), request(pickup_datetime=None)])


def test_temporal_extrapolation_uses_training_datetime_range(trained):
    bundle = trained["store"].load_current()
    inside = predict_trip(bundle, **request(pickup_datetime=pd.Timestamp("2016-03-15 08:30")))
    jan_2027 = predict_trip(bundle, **request(pickup_datetime=pd.Timestamp("2027-01-15 08:30")))
    assert inside["temporal_extrapolation"] is False
    assert jan_2027["temporal_extrapolation"] is True


def test_fare_rules():
    jfk = landmarks.get("jfk")
    rows = pd.DataFrame([
        {**request(), "pickup_datetime": pd.Timestamp("2016-03-15 08:30")},   # weekday rush
        {**request(), "pickup_datetime": pd.Timestamp("2016-03-19 08:30")},   # weekend same hour
        {**request(dropoff_latitude=jfk.lat, dropoff_longitude=jfk.lon), "pickup_datetime": pd.Timestamp("2016-03-16 02:00")},
    ])
    feats = build_features(rows)
    fares = fare_components(feats["distance_manhattan_km"].to_numpy(), np.array([600.0, 600.0, 2400.0]), feats)
    card = config.FARE_CARD
    assert fares.loc[0, "rush_hour_surcharge"] == card["rush_hour_surcharge"]
    assert fares.loc[1, "rush_hour_surcharge"] == 0
    assert fares.loc[2, "airport_fee"] == card["airport_fee"] and fares.loc[0, "airport_fee"] == 0
    assert fares.loc[2, "late_night_surcharge"] == card["late_night_surcharge"]
    assert fares.loc[0, "time_charge"] == pytest.approx(10 * card["per_minute"])
    assert (fares["congestion_fee"] == card["congestion_fee"]).all()


def test_retrain_hyperparameters_change_artifacts_and_predictions(trained, processed_path):
    store = trained["store"]
    before = store.load_current()
    v2 = run_training(TrainParams(seed=3, n_estimators=50, max_depth=10, learning_rate=0.3, subsample=0.6),
                      trips=TripStore(processed_path), store=store)
    after = store.load_current()
    assert after.version == v2 != before.version
    assert after.metadata["hyperparameters"]["max_depth"] == 10
    assert after.metadata["hyperparameters"]["subsample"] == 0.6
    assert predict_trip(after, **request())["duration"]["seconds"] != predict_trip(before, **request())["duration"]["seconds"]
    assert after.diagnostics["deep_dive"]["residuals"] != before.diagnostics["deep_dive"]["residuals"]
    # Validation population is identical across the two versions.
    assert after.metadata["rows"]["validation"] == before.metadata["rows"]["validation"]


def test_retrain_validation_rejects_out_of_range_params():
    with pytest.raises(ValueError, match="max_depth"):
        TrainParams(max_depth=20).validate()
    with pytest.raises(ValueError, match="sample_size"):
        TrainParams(sample_size=100).validate()


def test_failed_promotion_leaves_pointer_unchanged(trained, processed_path, monkeypatch):
    store = trained["store"]
    current = store.current_version()
    n_versions = len(store.list_versions())

    def boom(version):
        raise RuntimeError("validation failed")
    monkeypatch.setattr(store, "validate_version", boom)
    with pytest.raises(RuntimeError, match="validation failed"):
        run_training(TrainParams(seed=9, **FAST_PARAMS), trips=TripStore(processed_path), store=store)
    assert store.current_version() == current
    assert len(store.list_versions()) == n_versions


def test_finalize_once_then_refuse_unless_forced(trained, processed_path):
    store = trained["store"]
    trips = TripStore(processed_path)
    result = holdout.finalize(trips=trips, store=store)
    assert "test" in trips.accessed
    assert result["metrics"]["n"] == len(trips.role("test"))
    assert store.load_current().holdout is not None
    with pytest.raises(holdout.HoldoutAlreadyEvaluated):
        holdout.finalize(trips=TripStore(processed_path), store=store)
    forced = holdout.finalize(force=True, trips=TripStore(processed_path), store=store)
    assert forced["forced_rerun"] is True
    log = store.read_holdout_log()
    assert [e["forced_rerun"] for e in log] == [False, True]


def test_new_version_is_not_finalized(trained, processed_path):
    store = trained["store"]
    run_training(TrainParams(seed=11, **FAST_PARAMS), trips=TripStore(processed_path), store=store)
    assert store.load_current().holdout is None


def test_report_reflects_live_artifacts(trained):
    from taxi.report import PHASES, build_report
    bundle = trained["store"].load_current()
    rep = build_report(bundle.metadata, bundle.experiments, bundle.diagnostics, bundle.holdout, None,
                       trained["trips"].report())
    assert [k for k, _ in PHASES] == list(rep)
    text = " ".join(v for blocks in rep.values() for kind, v in blocks if kind == "md")
    assert f"{bundle.metadata['validation_metrics']['rmsle']:.4f}" in text
    if bundle.holdout is None:
        assert "not been finalized" in text
