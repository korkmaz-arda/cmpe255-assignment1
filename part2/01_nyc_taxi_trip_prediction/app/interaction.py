"""Framework-free estimator interaction rules (map events, landmark matching, time presets).

No Streamlit import, so these rules are unit-tested directly. Spatial *features* are never computed
here: coordinates go unchanged to `taxi.features.build_features` via `taxi.predict`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time

from taxi import config, landmarks

ENDPOINTS = ("pickup", "dropoff")
CUSTOM = "__custom__"

# Convenience presets; each maps to one exact clock time (the user can still type any minute).
TIME_PRESETS: dict[str, time] = {
    "03:00 · late night": time(3, 0),
    "08:30 · morning rush": time(8, 30),
    "12:00 · midday": time(12, 0),
    "17:30 · evening rush": time(17, 30),
    "22:00 · evening": time(22, 0),
}


# Selectable ride dates. Streamlit's default window is only +/-10 years around the initial value, which would
# exclude the 2016 training period; these explicit bounds cover it and leave room for future-date what-ifs.
DATE_MIN = date(2009, 1, 1)
DATE_MAX = date(2035, 12, 31)
TRAINING_EXAMPLE_DATE = date(2016, 3, 15)  # a Tuesday inside the Jan-Jun 2016 training period


# Fixed estimator defaults on app start: 2026-01-01 at 12:00 AM (midnight).
DEFAULT_RIDE_DATE = date(2026, 1, 1)
DEFAULT_PICKUP_TIME = time(0, 0)


@dataclass(frozen=True)
class EndpointUpdate:
    endpoint: str
    lat: float
    lon: float
    choice: str  # landmark id when the point is exactly a landmark, else CUSTOM


def in_service_area(lat: float, lon: float) -> bool:
    return config.LAT_MIN <= lat <= config.LAT_MAX and config.LON_MIN <= lon <= config.LON_MAX


def match_landmark(lat: float, lon: float, tol: float = 1e-9) -> str:
    """Landmark id if (lat, lon) is exactly that landmark's coordinates, otherwise CUSTOM."""
    for lm in landmarks.LANDMARKS:
        if abs(lm.lat - lat) <= tol and abs(lm.lon - lon) <= tol:
            return lm.id
    return CUSTOM


def _coord(value, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Map event has no valid {name}") from exc
    if not math.isfinite(v):
        raise ValueError(f"Map event has a non-finite {name}")
    return v


def apply_map_event(event: dict, mode: str) -> EndpointUpdate:
    """Translate one map event into an update of exactly one endpoint.

    click     -> the endpoint selected by `mode` moves to the clicked coordinates
    landmark  -> the endpoint selected by `mode` snaps to that landmark
    drag      -> the dragged marker's own endpoint moves (independent of mode)
    Raises ValueError for malformed events or points outside the NYC service area.
    """
    if mode not in ENDPOINTS:
        raise ValueError(f"Unknown map mode {mode!r}")
    if not isinstance(event, dict):
        raise ValueError("Malformed map event")
    kind = event.get("type")
    if kind == "landmark":
        lm = landmarks.get(event.get("id"))
        return EndpointUpdate(mode, lm.lat, lm.lon, lm.id)
    if kind in ("click", "drag"):
        endpoint = mode if kind == "click" else event.get("endpoint")
        if endpoint not in ENDPOINTS:
            raise ValueError(f"Drag event names unknown endpoint {endpoint!r}")
        lat, lon = _coord(event.get("lat"), "latitude"), _coord(event.get("lng"), "longitude")
        if not in_service_area(lat, lon):
            raise ValueError(f"({lat:.5f}, {lon:.5f}) is outside the NYC service area the model covers")
        return EndpointUpdate(endpoint, lat, lon, match_landmark(lat, lon))
    raise ValueError(f"Unknown map event type {kind!r}")


def combine_pickup(ride_date: date | None, pickup_time: time | None) -> datetime | None:
    """Explicit date + explicit time -> the single pickup datetime the model receives."""
    if ride_date is None or pickup_time is None:
        return None
    return datetime.combine(ride_date, pickup_time.replace(second=0, microsecond=0))
