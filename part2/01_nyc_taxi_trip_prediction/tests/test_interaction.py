"""Estimator interaction rules: map events, landmark matching, time handling (no Streamlit runtime)."""
from datetime import date, datetime, time

import pytest

from app.interaction import (CUSTOM, TIME_PRESETS, apply_map_event, combine_pickup, in_service_area,
                             match_landmark)
from taxi import landmarks


def test_click_moves_only_the_endpoint_selected_by_mode():
    ev = {"type": "click", "lat": 40.712345678, "lng": -73.998765432, "seq": "1"}
    up = apply_map_event(ev, "pickup")
    assert (up.endpoint, up.lat, up.lon, up.choice) == ("pickup", 40.712345678, -73.998765432, CUSTOM)
    assert apply_map_event(ev, "dropoff").endpoint == "dropoff"


def test_click_coordinates_are_passed_through_exactly():
    lat, lng = 40.74831234567891, -73.98561234567891
    up = apply_map_event({"type": "click", "lat": lat, "lng": lng}, "dropoff")
    assert up.lat == lat and up.lon == lng


def test_drag_moves_the_dragged_marker_regardless_of_mode():
    up = apply_map_event({"type": "drag", "endpoint": "pickup", "lat": 40.75, "lng": -73.99}, "dropoff")
    assert up.endpoint == "pickup"


def test_landmark_event_snaps_mode_endpoint_to_exact_landmark():
    jfk = landmarks.get("jfk")
    up = apply_map_event({"type": "landmark", "id": "jfk"}, "pickup")
    assert (up.endpoint, up.lat, up.lon, up.choice) == ("pickup", jfk.lat, jfk.lon, "jfk")


@pytest.mark.parametrize("event", [
    {"type": "click", "lat": 51.5, "lng": -0.12},                  # outside the service area
    {"type": "click", "lat": float("nan"), "lng": -73.9},
    {"type": "click", "lat": None, "lng": -73.9},
    {"type": "drag", "endpoint": "both", "lat": 40.75, "lng": -73.99},
    {"type": "teleport", "lat": 40.75, "lng": -73.99},
    "not-a-dict",
])
def test_invalid_map_events_are_rejected(event):
    with pytest.raises((ValueError, KeyError)):
        apply_map_event(event, "pickup")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        apply_map_event({"type": "click", "lat": 40.75, "lng": -73.99}, "both")


def test_landmark_label_kept_only_at_exact_coordinates():
    ts = landmarks.get("times_square")
    assert match_landmark(ts.lat, ts.lon) == "times_square"
    assert match_landmark(ts.lat + 1e-6, ts.lon) == CUSTOM
    for lm in landmarks.LANDMARKS:
        assert in_service_area(lm.lat, lm.lon)


def test_time_presets_map_to_exact_times():
    assert TIME_PRESETS["08:30 · morning rush"] == time(8, 30)
    assert TIME_PRESETS["17:30 · evening rush"] == time(17, 30)
    assert TIME_PRESETS["03:00 · late night"] == time(3, 0)
    assert all(t.second == 0 for t in TIME_PRESETS.values())


def test_combine_pickup_uses_explicit_date_and_time():
    assert combine_pickup(date(2016, 3, 15), time(17, 42, 31)) == datetime(2016, 3, 15, 17, 42)
    assert combine_pickup(None, time(8, 0)) is None
    assert combine_pickup(date(2016, 3, 15), None) is None


def test_date_bounds_include_training_period():
    from app.interaction import DATE_MAX, DATE_MIN, TRAINING_EXAMPLE_DATE
    assert DATE_MIN <= date(2016, 1, 1) and date(2016, 6, 30) <= DATE_MAX
    assert TRAINING_EXAMPLE_DATE.weekday() == 1 and TRAINING_EXAMPLE_DATE.year == 2016
