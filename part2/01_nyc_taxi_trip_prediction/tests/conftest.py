"""Shared offline fixtures: a small *synthetic* trip table in the Kaggle schema (no Kaggle data)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from taxi import data  # noqa: E402
from taxi.features import manhattan_km  # noqa: E402


def make_synthetic_trips(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    plat = rng.uniform(40.70, 40.80, n)
    plon = rng.uniform(-74.01, -73.93, n)
    dlat = rng.uniform(40.70, 40.80, n)
    dlon = rng.uniform(-74.01, -73.93, n)
    start = pd.Timestamp("2016-01-01")
    pickup = start + pd.to_timedelta(rng.uniform(0, 181 * 86400, n), unit="s")
    pickup = pickup.floor("s")
    dist = manhattan_km(plat, plon, dlat, dlon)
    hour = pickup.hour.to_numpy()
    slow = np.isin(hour, [7, 8, 9, 16, 17, 18, 19]) & (pickup.dayofweek.to_numpy() < 5)
    duration = (120 + dist * np.where(slow, 260, 170) * rng.lognormal(0, 0.2, n)).round().astype(int)
    return pd.DataFrame({
        "id": [f"id{i:07d}" for i in range(n)],
        "vendor_id": rng.integers(1, 3, n),
        "pickup_datetime": pickup,
        "dropoff_datetime": pickup + pd.to_timedelta(duration, unit="s"),
        "passenger_count": rng.choice([1, 1, 1, 2, 3, 5], n),
        "pickup_longitude": plon, "pickup_latitude": plat,
        "dropoff_longitude": dlon, "dropoff_latitude": dlat,
        "store_and_fwd_flag": "N",
        "trip_duration": duration,
    })


def write_processed(df: pd.DataFrame, directory: Path) -> Path:
    cleaned, counts = data.clean(df)
    cleaned["split"] = data.assign_split(cleaned["id"])
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "trips_clean.parquet"
    cleaned.to_parquet(path, index=False)
    report = {"raw_rows": len(df), "cleaning": counts, "clean_rows": len(cleaned),
              "split_sizes": {r: int((cleaned["split"] == r).sum()) for r in data.ROLES},
              "pickup_datetime_min": cleaned["pickup_datetime"].min().isoformat(),
              "pickup_datetime_max": cleaned["pickup_datetime"].max().isoformat()}
    (directory / "data_report.json").write_text(json.dumps(report))
    return path


@pytest.fixture(scope="session")
def synthetic_trips() -> pd.DataFrame:
    return make_synthetic_trips()


@pytest.fixture(scope="session")
def processed_path(tmp_path_factory, synthetic_trips) -> Path:
    return write_processed(synthetic_trips, tmp_path_factory.mktemp("processed"))


FAST_PARAMS = dict(n_estimators=60, max_depth=4, learning_rate=0.2, subsample=0.9)
