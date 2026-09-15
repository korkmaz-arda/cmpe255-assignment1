"""Single and batch trip predictions with band, illustrative fare, and route telemetry (F06/F07).

Inputs must be explicit: a missing pickup datetime or passenger count raises ValueError. Callers
(e.g. the UI form) own any defaults. Nothing here reads the clock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .artifacts import ModelBundle
from .diagnostics import band_offsets
from .features import build_features, distance_tier

FARE_LABELS = {
    "base_fare": "Base fare",
    "distance_charge": "Distance charge (city-block miles)",
    "time_charge": "Time charge (predicted minutes)",
    "rush_hour_surcharge": "Weekday rush-hour surcharge",
    "late_night_surcharge": "Late-night surcharge",
    "congestion_fee": "Congestion fee",
    "airport_fee": "Airport access fee (JFK/LGA)",
}


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60}m {total % 60:02d}s"


def _check_domain(req: pd.DataFrame) -> None:
    pc = pd.to_numeric(req["passenger_count"], errors="coerce")
    bad_pc = req.index[~pc.between(config.MIN_PASSENGERS, config.MAX_PASSENGERS)]
    if len(bad_pc):
        raise ValueError(f"passenger_count must be {config.MIN_PASSENGERS}-{config.MAX_PASSENGERS}; "
                         f"invalid in row(s): {', '.join(map(str, bad_pc[:10]))}")
    for prefix in ("pickup", "dropoff"):
        lat, lon = req[f"{prefix}_latitude"].astype(float), req[f"{prefix}_longitude"].astype(float)
        outside = req.index[~(lat.between(config.LAT_MIN, config.LAT_MAX) & lon.between(config.LON_MIN, config.LON_MAX))]
        if len(outside):
            raise ValueError(f"{prefix} coordinates outside the NYC service area in row(s): "
                             f"{', '.join(map(str, outside[:10]))}")
    try:
        pd.to_datetime(req["pickup_datetime"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"pickup_datetime could not be parsed: {exc}") from exc


def fare_components(manhattan_km: np.ndarray, duration_s: np.ndarray, features: pd.DataFrame) -> pd.DataFrame:
    card = config.FARE_CARD
    n = len(features)
    near_airport = ((features["dist_jfk_km"] <= config.AIRPORT_RADIUS_KM)
                    | (features["dist_lga_km"] <= config.AIRPORT_RADIUS_KM)).to_numpy()
    comp = pd.DataFrame({
        "base_fare": np.full(n, card["base_fare"]),
        "distance_charge": manhattan_km / config.KM_PER_MILE * card["per_mile"],
        "time_charge": duration_s / 60.0 * card["per_minute"],
        "rush_hour_surcharge": features["is_rush_hour"].to_numpy() * card["rush_hour_surcharge"],
        "late_night_surcharge": features["is_late_night"].to_numpy() * card["late_night_surcharge"],
        "congestion_fee": np.full(n, card["congestion_fee"]),
        "airport_fee": near_airport * card["airport_fee"],
    }, index=features.index)
    return comp.round(2)


def predict_frame(bundle: ModelBundle, trips: pd.DataFrame) -> pd.DataFrame:
    """Vectorised predictions; one output row per input trip (same index)."""
    feats = build_features(trips)  # raises on missing request fields
    _check_domain(trips[config.REQUEST_COLUMNS])
    pred_log = bundle.predict_log(feats)
    duration = np.maximum(np.expm1(pred_log), config.MIN_PREDICTED_DURATION_S)
    tiers = distance_tier(feats)
    low_off, high_off = band_offsets(bundle.metadata["band"], tiers)
    band_low = np.maximum(np.expm1(pred_log + low_off), config.MIN_PREDICTED_DURATION_S)
    band_high = np.maximum(np.expm1(pred_log + high_off), band_low)
    fares = fare_components(feats["distance_manhattan_km"].to_numpy(), duration, feats)

    rng = bundle.metadata["training_pickup_range"]
    ts = pd.to_datetime(trips["pickup_datetime"])
    extrapolation = ((ts < pd.Timestamp(rng["min"])) | (ts > pd.Timestamp(rng["max"]))).to_numpy()

    out = pd.DataFrame({
        "duration_seconds": duration,
        "duration_minutes": duration / 60.0,
        "duration_formatted": [format_duration(s) for s in duration],
        "band_low_seconds": band_low,
        "band_high_seconds": band_high,
        "band_tier": tiers,
        "distance_haversine_km": feats["distance_haversine_km"].to_numpy(),
        "distance_manhattan_km": feats["distance_manhattan_km"].to_numpy(),
        "bearing_deg": feats["bearing_deg"].to_numpy(),
        "avg_speed_kmh": feats["distance_manhattan_km"].to_numpy() / (duration / 3600.0),
        "temporal_extrapolation": extrapolation,
    }, index=trips.index)
    out["distance_haversine_mi"] = out["distance_haversine_km"] / config.KM_PER_MILE
    out["distance_manhattan_mi"] = out["distance_manhattan_km"] / config.KM_PER_MILE
    out["avg_speed_mph"] = out["avg_speed_kmh"] / config.KM_PER_MILE
    for col in fares.columns:
        out[f"fare_{col}"] = fares[col]
    out["fare_total"] = fares.sum(axis=1).round(2)
    return out


def predict_trip(bundle: ModelBundle, *, pickup_latitude: float, pickup_longitude: float,
                 dropoff_latitude: float, dropoff_longitude: float, pickup_datetime,
                 passenger_count) -> dict:
    """Single-trip payload. All six request fields are required."""
    trip = pd.DataFrame([{
        "pickup_datetime": pickup_datetime, "passenger_count": passenger_count,
        "pickup_latitude": pickup_latitude, "pickup_longitude": pickup_longitude,
        "dropoff_latitude": dropoff_latitude, "dropoff_longitude": dropoff_longitude,
    }])
    row = predict_frame(bundle, trip).iloc[0]
    fare_items = [{"key": k, "label": FARE_LABELS[k], "amount": float(row[f"fare_{k}"])}
                  for k in FARE_LABELS]
    return {
        "duration": {"seconds": float(row["duration_seconds"]), "minutes": float(row["duration_minutes"]),
                     "formatted": row["duration_formatted"]},
        "band": {"low_seconds": float(row["band_low_seconds"]), "high_seconds": float(row["band_high_seconds"]),
                 "level": bundle.metadata["band"]["level"], "tier": row["band_tier"]},
        "fare": {"items": fare_items, "total": float(row["fare_total"]),
                 "note": "Illustrative rule-based estimate, not a TLC meter quote."},
        "route": {
            "distance_manhattan_km": float(row["distance_manhattan_km"]),
            "distance_manhattan_mi": float(row["distance_manhattan_mi"]),
            "distance_haversine_km": float(row["distance_haversine_km"]),
            "distance_haversine_mi": float(row["distance_haversine_mi"]),
            "bearing_deg": float(row["bearing_deg"]),
            "avg_speed_kmh": float(row["avg_speed_kmh"]),
            "avg_speed_mph": float(row["avg_speed_mph"]),
        },
        "temporal_extrapolation": bool(row["temporal_extrapolation"]),
        "training_pickup_range": bundle.metadata["training_pickup_range"],
        "model_version": bundle.version,
    }


def predict_batch(bundle: ModelBundle, trips: pd.DataFrame | list[dict]) -> dict:
    """Bulk predictions: per-trip rows plus a count (F07)."""
    df = pd.DataFrame(trips) if not isinstance(trips, pd.DataFrame) else trips.copy()
    if len(df) == 0:
        return {"count": 0, "predictions": pd.DataFrame()}
    df = df.reset_index(drop=True)
    return {"count": int(len(df)), "predictions": predict_frame(bundle, df)}
