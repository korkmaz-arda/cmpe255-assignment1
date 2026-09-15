"""Spatial-temporal feature engineering shared by training and inference (F03).

`build_features` is a pure function of request-time inputs only. It never reads the clock and
never reads post-trip fields (dropoff time, duration) or the row id - see config.REQUEST_COLUMNS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

EARTH_RADIUS_KM = 6371.0088

BASE_FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
    "distance_haversine_km",
    "distance_manhattan_km",
    "bearing_deg",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "is_rush_hour",
    "is_late_night",
    "passenger_count",
] + [f"dist_{key}_km" for key, *_ in config.HUBS]

FEATURE_DESCRIPTIONS = {
    "pickup_latitude": "Pickup latitude (degrees)",
    "pickup_longitude": "Pickup longitude (degrees)",
    "dropoff_latitude": "Dropoff latitude (degrees)",
    "dropoff_longitude": "Dropoff longitude (degrees)",
    "distance_haversine_km": "Great-circle (straight-line) distance, km",
    "distance_manhattan_km": "City-block distance: east-west leg + north-south leg, km",
    "bearing_deg": "Compass heading from pickup to dropoff, 0-360 degrees",
    "hour": "Pickup hour of day (0-23)",
    "day_of_week": "Pickup day of week (0 = Monday)",
    "month": "Pickup month (1-12)",
    "is_weekend": "1 if Saturday or Sunday",
    "is_rush_hour": "1 if a weekday and hour in 07-09 or 16-19",
    "is_late_night": "1 if hour in 00-05",
    "passenger_count": "Driver-reported passengers (1-6)",
    **{f"dist_{key}_km": f"Closest approach to {name}: min(pickup, dropoff) distance, km"
       for key, name, *_ in config.HUBS},
}


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def manhattan_km(lat1, lon1, lat2, lon2):
    """Sum of the pure-longitude leg (at pickup latitude) and the pure-latitude leg."""
    lat1 = np.asarray(lat1, dtype=float)
    east_west = haversine_km(lat1, lon1, lat1, lon2)
    north_south = haversine_km(lat1, lon1, lat2, lon1)
    return east_west + north_south


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(v, dtype=float)) for v in (lat1, lon1, lat2, lon2))
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def validate_request_frame(df: pd.DataFrame) -> None:
    """Raise ValueError when a required request-time field is absent or missing."""
    missing_cols = [c for c in config.REQUEST_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required input column(s): {', '.join(missing_cols)}")
    for col in config.REQUEST_COLUMNS:
        bad = df.index[df[col].isna()]
        if len(bad):
            shown = ", ".join(str(i) for i in bad[:10])
            raise ValueError(f"Missing value for '{col}' in row(s): {shown}"
                             + (" ..." if len(bad) > 10 else ""))
    for col in config.REQUEST_COLUMNS[1:]:  # numeric fields
        bad = df.index[pd.to_numeric(df[col], errors="coerce").isna()]
        if len(bad):
            shown = ", ".join(str(i) for i in bad[:10])
            raise ValueError(f"Non-numeric value for '{col}' in row(s): {shown}"
                             + (" ..." if len(bad) > 10 else ""))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the fixed, ordered 20-column feature matrix for the given trips."""
    validate_request_frame(df)
    req = df[config.REQUEST_COLUMNS]  # the only columns read from here on
    ts = pd.to_datetime(req["pickup_datetime"])
    plat = req["pickup_latitude"].to_numpy(dtype=float)
    plon = req["pickup_longitude"].to_numpy(dtype=float)
    dlat = req["dropoff_latitude"].to_numpy(dtype=float)
    dlon = req["dropoff_longitude"].to_numpy(dtype=float)

    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    is_weekday = dow < 5

    out = {
        "pickup_latitude": plat,
        "pickup_longitude": plon,
        "dropoff_latitude": dlat,
        "dropoff_longitude": dlon,
        "distance_haversine_km": haversine_km(plat, plon, dlat, dlon),
        "distance_manhattan_km": manhattan_km(plat, plon, dlat, dlon),
        "bearing_deg": bearing_deg(plat, plon, dlat, dlon),
        "hour": hour,
        "day_of_week": dow,
        "month": ts.dt.month.to_numpy(),
        "is_weekend": (~is_weekday).astype(int),
        "is_rush_hour": (is_weekday & np.isin(hour, list(config.RUSH_HOURS))).astype(int),
        "is_late_night": np.isin(hour, list(config.LATE_NIGHT_HOURS)).astype(int),
        "passenger_count": req["passenger_count"].to_numpy(dtype=float),
    }
    for key, _name, hlat, hlon in config.HUBS:
        out[f"dist_{key}_km"] = np.minimum(haversine_km(plat, plon, hlat, hlon),
                                           haversine_km(dlat, dlon, hlat, hlon))
    feats = pd.DataFrame(out, index=df.index)
    return feats[BASE_FEATURES].astype(float)


def distance_tier(features: pd.DataFrame) -> np.ndarray:
    """Non-overlapping error tiers; the airport tier takes precedence."""
    near_airport = (
        (features["dist_jfk_km"] <= config.AIRPORT_RADIUS_KM)
        | (features["dist_lga_km"] <= config.AIRPORT_RADIUS_KM)
        | (features["dist_ewr_km"] <= config.AIRPORT_RADIUS_KM)
    ).to_numpy()
    d = features["distance_haversine_km"].to_numpy()
    tier = np.where(d < 2.0, "short", np.where(d < 5.0, "commute", "cross_borough"))
    return np.where(near_airport, "airport", tier).astype(object)


TIER_LABELS = {
    "short": "Short (< 2 km)",
    "commute": "Commute (2–5 km)",
    "cross_borough": "Long / cross-borough (≥ 5 km)",
    "airport": "Airport (JFK/LGA/EWR endpoint)",
}
TIER_ORDER = ["short", "commute", "cross_borough", "airport"]
