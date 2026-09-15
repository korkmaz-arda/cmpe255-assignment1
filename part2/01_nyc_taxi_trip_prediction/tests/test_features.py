import numpy as np
import pandas as pd
import pytest

from taxi import config, landmarks
from taxi.features import (BASE_FEATURES, bearing_deg, build_features, distance_tier, haversine_km,
                           manhattan_km)


def trip(**over):
    row = {"pickup_datetime": pd.Timestamp("2016-03-15 08:30"), "passenger_count": 2,
           "pickup_latitude": 40.7580, "pickup_longitude": -73.9855,
           "dropoff_latitude": 40.7060, "dropoff_longitude": -74.0086}
    row.update(over)
    return pd.DataFrame([row])


def test_haversine_known_values():
    # One degree of longitude on the equator ~ 111.195 km with the mean Earth radius.
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.195, abs=0.01)
    assert haversine_km(40.7, -74.0, 40.7, -74.0) == pytest.approx(0.0)


def test_manhattan_is_sum_of_orthogonal_legs():
    lat1, lon1, lat2, lon2 = 40.75, -73.99, 40.70, -73.95
    expected = haversine_km(lat1, lon1, lat1, lon2) + haversine_km(lat1, lon1, lat2, lon1)
    assert manhattan_km(lat1, lon1, lat2, lon2) == pytest.approx(expected)
    assert manhattan_km(lat1, lon1, lat2, lon2) >= haversine_km(lat1, lon1, lat2, lon2)


def test_bearing_cardinal_directions():
    assert bearing_deg(40.0, -74.0, 41.0, -74.0) == pytest.approx(0.0, abs=1e-6)
    assert bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=1e-6)
    assert bearing_deg(41.0, -74.0, 40.0, -74.0) == pytest.approx(180.0, abs=1e-6)


def test_feature_matrix_has_fixed_20_columns_in_order():
    X = build_features(trip())
    assert list(X.columns) == BASE_FEATURES
    assert len(BASE_FEATURES) == 20


def test_rush_hour_is_weekday_only():
    tuesday = build_features(trip(pickup_datetime=pd.Timestamp("2016-03-15 08:30")))
    saturday = build_features(trip(pickup_datetime=pd.Timestamp("2016-03-19 08:30")))
    assert tuesday["is_rush_hour"].iloc[0] == 1
    assert saturday["is_rush_hour"].iloc[0] == 0
    assert saturday["is_weekend"].iloc[0] == 1
    late = build_features(trip(pickup_datetime=pd.Timestamp("2016-03-19 03:10")))
    assert late["is_late_night"].iloc[0] == 1


def test_hub_proximity_is_min_of_pickup_and_dropoff():
    jfk = landmarks.get("jfk")
    X = build_features(trip(dropoff_latitude=jfk.lat, dropoff_longitude=jfk.lon))
    assert X["dist_jfk_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    # Times Square pickup -> hub distance comes from the pickup end.
    assert X["dist_times_sq_km"].iloc[0] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("field", ["pickup_datetime", "passenger_count", "pickup_latitude"])
def test_missing_request_fields_raise(field):
    with pytest.raises(ValueError, match=field):
        build_features(trip(**{field: None}))
    with pytest.raises(ValueError, match=field):
        build_features(trip().drop(columns=[field]))


def test_no_post_trip_or_identity_fields_are_features():
    forbidden = set(config.POST_TRIP_COLUMNS) | {config.ID_COLUMN, "dropoff_datetime", "trip_duration"}
    assert not forbidden & set(BASE_FEATURES)
    assert not any("duration" in f or "dropoff_datetime" in f or f == "id" for f in BASE_FEATURES)


def test_post_trip_fields_cannot_change_feature_vector():
    base = trip()
    a = base.assign(id="id1", dropoff_datetime=pd.Timestamp("2016-03-15 08:40"), trip_duration=600)
    b = base.assign(id="zzz", dropoff_datetime=pd.Timestamp("2016-03-15 11:00"), trip_duration=9000)
    pd.testing.assert_frame_equal(build_features(a), build_features(b))


def test_build_features_is_deterministic():
    pd.testing.assert_frame_equal(build_features(trip()), build_features(trip()))


def test_distance_tiers_do_not_overlap_and_airport_wins():
    jfk = landmarks.get("jfk")
    rows = pd.concat([
        trip(dropoff_latitude=40.7585, dropoff_longitude=-73.9850),  # short
        trip(),                                                      # ~6 km -> cross_borough
        trip(dropoff_latitude=jfk.lat, dropoff_longitude=jfk.lon),   # airport
    ], ignore_index=True)
    tiers = distance_tier(build_features(rows))
    assert list(tiers) == ["short", "cross_borough", "airport"]


def test_landmark_catalogue_ids_unique_and_quick_picks_valid():
    ids = landmarks.ids()
    assert len(ids) == len(set(ids)) == 10
    for q in landmarks.QUICK_PICKS:
        assert landmarks.get(q).id == q
    for lm in landmarks.LANDMARKS:
        assert config.LAT_MIN <= lm.lat <= config.LAT_MAX and config.LON_MIN <= lm.lon <= config.LON_MAX


def test_non_numeric_request_fields_raise_specific_error():
    # Regression (final audit): a text coordinate previously surfaced only as "could not convert string to float".
    with pytest.raises(ValueError, match="Non-numeric value for 'pickup_latitude' in row\\(s\\): 0"):
        build_features(trip(pickup_latitude="abc"))
    with pytest.raises(ValueError, match="Non-numeric value for 'passenger_count'"):
        build_features(trip(passenger_count="two"))
