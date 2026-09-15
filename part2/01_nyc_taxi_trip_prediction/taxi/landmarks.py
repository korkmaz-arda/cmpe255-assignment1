"""The single authoritative landmark catalogue (S02).

Selectors, quick-select chips and map nodes all resolve landmarks through this module by id.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Landmark:
    id: str
    name: str
    lat: float
    lon: float
    zone: str
    icon: str


LANDMARKS: tuple[Landmark, ...] = (
    Landmark("times_square", "Times Square", 40.7580, -73.9855, "Midtown Manhattan", "🎭"),
    Landmark("jfk", "JFK Airport", 40.6413, -73.7781, "Queens · JFK", "✈️"),
    Landmark("lga", "LaGuardia Airport", 40.7769, -73.8740, "Queens · LGA", "🛫"),
    Landmark("central_park", "Central Park", 40.7812, -73.9665, "Upper Manhattan", "🌳"),
    Landmark("wall_street", "Wall Street", 40.7060, -74.0086, "Lower Manhattan", "🏦"),
    Landmark("brooklyn_bridge", "Brooklyn Bridge", 40.7061, -73.9969, "Lower Manhattan", "🌉"),
    Landmark("grand_central", "Grand Central", 40.7527, -73.9772, "Midtown Manhattan", "🚉"),
    Landmark("empire_state", "Empire State Building", 40.7484, -73.9857, "Midtown Manhattan", "🏙️"),
    Landmark("williamsburg", "Williamsburg", 40.7081, -73.9571, "Brooklyn", "🎨"),
    Landmark("astoria", "Astoria", 40.7644, -73.9235, "Queens", "🏘️"),
)

_BY_ID = {lm.id: lm for lm in LANDMARKS}
QUICK_PICKS = ("times_square", "jfk", "lga", "wall_street")


def get(landmark_id: str) -> Landmark:
    try:
        return _BY_ID[landmark_id]
    except KeyError as exc:
        raise KeyError(f"Unknown landmark id: {landmark_id!r}") from exc


def ids() -> list[str]:
    return [lm.id for lm in LANDMARKS]
