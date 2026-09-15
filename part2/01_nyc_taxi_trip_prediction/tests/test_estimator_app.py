"""Headless AppTest runs of the real estimator against a model trained on synthetic fixtures."""
import sys
from datetime import date, time
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from taxi import config, landmarks
from taxi.artifacts import ArtifactStore
from taxi.data import TripStore
from taxi.train import TrainParams, run_training

from conftest import FAST_PARAMS

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(scope="module")
def artifacts_root(tmp_path_factory, processed_path):
    root = tmp_path_factory.mktemp("app_artifacts")
    run_training(TrainParams(seed=2, **FAST_PARAMS), trips=TripStore(processed_path), store=ArtifactStore(root))
    return root


@pytest.fixture
def app(monkeypatch, artifacts_root, processed_path):
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_root)
    monkeypatch.setattr(config, "CLEAN_PARQUET", processed_path)
    # Each AppTest has a fresh component registry; re-import app modules so the map component registers again.
    for name in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        monkeypatch.delitem(sys.modules, name)
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    return at


def duration_text(at) -> str:
    for md in at.markdown:
        if "Predicted duration" in md.value:
            return md.value.split("class='hero'>")[1].split("<")[0]
    raise AssertionError("no prediction rendered")


def readout(at) -> str:
    return next(md.value for md in at.markdown if "Model receives" in md.value)


def set_date(at, d):
    at.date_input(key="ride_date").set_value(d)


def set_when(at, d, t):
    set_date(at, d)
    at.time_input(key="pickup_time").set_value(t)
    at.run()
    assert not at.exception


def test_date_and_time_are_separate_controls_that_combine(app):
    set_when(app, date(2016, 3, 15), time(17, 42))
    assert "2016-03-15 17:42" in readout(app)
    assert "Tuesday" in readout(app) and "hour 17" in readout(app) and "weekday rush hour" in readout(app)


def test_changing_only_the_hour_changes_features_and_prediction(app):
    set_when(app, date(2016, 3, 15), time(8, 30))
    rush = duration_text(app)
    assert "weekday rush hour" in readout(app)
    app.time_input(key="pickup_time").set_value(time(3, 0)).run()
    assert "late night" in readout(app) and "hour 3" in readout(app)
    assert "2016-03-15" in readout(app)          # date untouched
    assert duration_text(app) != rush


def test_changing_only_the_date_follows_weekday_features(app):
    set_when(app, date(2016, 3, 15), time(8, 30))   # Tuesday
    assert "weekday rush hour" in readout(app)
    set_date(app, date(2016, 3, 19))   # Saturday, same time
    app.run()
    r = readout(app)
    assert "Saturday" in r and "weekend" in r and "weekday rush hour" not in r and "08:30" in r


def test_date_widget_accepts_training_period_dates(app):
    # Regression: Streamlit's default +/-10-year window made every 2016 date unselectable.
    set_when(app, date(2016, 1, 1), time(9, 0))
    assert "2016-01-01 09:00" in readout(app)
    app.date_input(key="ride_date").set_value(date(2026, 1, 1)).run()
    app.button(key="date_training").click().run()
    assert app.date_input(key="ride_date").value == date(2016, 3, 15)
    assert app.time_input(key="pickup_time").value == time(9, 0)


def test_time_preset_and_shift_buttons_set_exact_times(app):
    app.pills(key="time_preset").set_value("17:30 · evening rush").run()
    assert app.time_input(key="pickup_time").value == time(17, 30)
    app.button(key="t_p15").click().run()
    assert app.time_input(key="pickup_time").value == time(17, 45)
    app.button(key="t_m60").click().run()
    assert app.time_input(key="pickup_time").value == time(16, 45)


def test_extrapolation_warning_uses_full_datetime(app):
    set_when(app, date(2016, 3, 15), time(8, 30))
    assert not any("Temporal extrapolation" in w.value for w in app.warning)
    set_when(app, date(2027, 1, 15), time(8, 30))     # January, but outside the 2016 training period
    assert any("Temporal extrapolation" in w.value for w in app.warning)


def test_coordinates_drive_prediction_and_landmark_label(app):
    set_when(app, date(2016, 3, 15), time(12, 0))
    before = duration_text(app)
    app.number_input(key="dropoff_lat").set_value(40.7600).run()
    app.number_input(key="dropoff_lon").set_value(-73.9800).run()
    assert app.selectbox(key="dropoff_choice").value == "__custom__"
    assert app.selectbox(key="pickup_choice").value == "times_square"     # other endpoint untouched
    assert duration_text(app) != before


def test_landmark_shortcuts_move_coordinates(app):
    app.selectbox(key="pickup_choice").set_value("wall_street").run()
    wall = landmarks.get("wall_street")
    assert app.number_input(key="pickup_lat").value == pytest.approx(wall.lat)
    assert app.number_input(key="pickup_lon").value == pytest.approx(wall.lon)
    app.pills(key="chip_dropoff").set_value("lga").run()
    lga = landmarks.get("lga")
    assert app.selectbox(key="dropoff_choice").value == "lga"
    assert app.number_input(key="dropoff_lat").value == pytest.approx(lga.lat)
    assert app.pills(key="chip_dropoff").value is None
    assert app.radio(key="map_mode").value == "dropoff"   # shortcuts never change the click mode


def test_app_starts_at_fixed_default_date_and_time(app):
    # Requested default: 2026-01-01 at 12:00 AM (midnight), not the current clock.
    assert app.date_input(key="ride_date").value == date(2026, 1, 1)
    assert app.time_input(key="pickup_time").value == time(0, 0)
    assert "2026-01-01 00:00" in readout(app) and "late night" in readout(app)
    assert any("Temporal extrapolation" in w.value for w in app.warning)   # 2026 lies outside Jan–Jun 2016
