"""Data acquisition from the local Kaggle archive, domain cleaning, and deterministic splits.

Split roles (by stable hash of the Kaggle trip `id`):
  fit          -> model fitting (retraining samples are drawn only from here)
  calibration  -> prediction-band residual quantiles only
  validation   -> early stopping, comparisons, AutoResearch, diagnostics (always the full split)
  test         -> one-time final holdout evaluation (taxi.holdout only)
"""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config
from .features import haversine_km

KEEP_COLUMNS = [
    "id", "vendor_id", "pickup_datetime", "dropoff_datetime", "passenger_count",
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "store_and_fwd_flag", "trip_duration",
]
ROLES = [role for role, *_ in config.SPLIT_BOUNDS]


# --- Acquisition -------------------------------------------------------------
def extract_archive(archive: Path = config.KAGGLE_ARCHIVE, out_dir: Path = config.RAW_DIR) -> dict[str, Path]:
    """Extract train.csv / test.csv from the Kaggle archive (which nests one zip per file)."""
    if not archive.exists():
        raise FileNotFoundError(
            f"Kaggle archive not found at {archive}. Place nyc-taxi-trip-duration.zip there (see README)."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with zipfile.ZipFile(archive) as outer:
        names = outer.namelist()
        for stem in ("train", "test"):
            target = out_dir / f"{stem}.csv"
            if target.exists():
                extracted[stem] = target
                continue
            if f"{stem}.csv" in names:
                with outer.open(f"{stem}.csv") as src, open(target, "wb") as dst:
                    dst.write(src.read())
            elif f"{stem}.zip" in names:
                with zipfile.ZipFile(io.BytesIO(outer.read(f"{stem}.zip"))) as inner, \
                        inner.open(f"{stem}.csv") as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
            else:
                raise FileNotFoundError(f"{stem}.csv/{stem}.zip not found inside {archive}")
            extracted[stem] = target
    return extracted


def load_raw_train(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=KEEP_COLUMNS)
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    return df


# --- Cleaning (F02) ----------------------------------------------------------
def _in_bbox(lat: pd.Series, lon: pd.Series) -> pd.Series:
    return lat.between(config.LAT_MIN, config.LAT_MAX) & lon.between(config.LON_MIN, config.LON_MAX)


def cleaning_rules() -> list[tuple[str, str, callable]]:
    """(key, description, predicate-that-keeps-valid-rows) applied sequentially."""
    return [
        ("duration_bounds",
         f"Trip duration between {config.MIN_DURATION_S} s and {config.MAX_DURATION_S // 3600} h",
         lambda d: d["trip_duration"].between(config.MIN_DURATION_S, config.MAX_DURATION_S)),
        ("pickup_in_nyc_bbox", "Pickup coordinates inside the NYC bounding box (incl. Newark)",
         lambda d: _in_bbox(d["pickup_latitude"], d["pickup_longitude"])),
        ("dropoff_in_nyc_bbox", "Dropoff coordinates inside the NYC bounding box (incl. Newark)",
         lambda d: _in_bbox(d["dropoff_latitude"], d["dropoff_longitude"])),
        ("passenger_count_range",
         f"Passenger count between {config.MIN_PASSENGERS} and {config.MAX_PASSENGERS}",
         lambda d: d["passenger_count"].between(config.MIN_PASSENGERS, config.MAX_PASSENGERS)),
        ("implied_speed",
         f"Great-circle speed at most {config.MAX_IMPLIED_SPEED_KMH:.0f} km/h (GPS/clock faults)",
         lambda d: haversine_km(d["pickup_latitude"], d["pickup_longitude"],
                                d["dropoff_latitude"], d["dropoff_longitude"])
         / (d["trip_duration"] / 3600.0) <= config.MAX_IMPLIED_SPEED_KMH),
    ]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply domain filters sequentially; return cleaned frame and per-rule removal counts."""
    counts = []
    for key, description, keep in cleaning_rules():
        mask = np.asarray(keep(df), dtype=bool)
        counts.append({"rule": key, "description": description,
                       "rows_before": int(len(df)), "removed": int((~mask).sum())})
        df = df.loc[mask]
    return df.reset_index(drop=True), counts


# --- Deterministic split -----------------------------------------------------
def hash_bucket(ids: pd.Series | list[str], salt: str = config.SPLIT_SALT) -> np.ndarray:
    """Per-mille bucket from md5(salt + id): stable across runs, row order, and library versions."""
    prefix = salt.encode() + b"|"
    return np.fromiter(
        (int.from_bytes(hashlib.md5(prefix + str(i).encode()).digest()[:4], "big") % 1000 for i in ids),
        dtype=np.int32, count=len(ids),
    )


def assign_split(ids: pd.Series | list[str]) -> np.ndarray:
    bucket = hash_bucket(ids)
    role = np.empty(len(bucket), dtype=object)
    for name, lo, hi in config.SPLIT_BOUNDS:
        role[(bucket >= lo) & (bucket < hi)] = name
    return role


def prepare(archive: Path = config.KAGGLE_ARCHIVE, raw_dir: Path = config.RAW_DIR,
            processed_dir: Path = config.PROCESSED_DIR) -> dict:
    """Extract, clean, split, and persist the modelling table plus a data report."""
    paths = extract_archive(archive, raw_dir)
    raw = load_raw_train(paths["train"])
    raw_rows = len(raw)
    cleaned, rule_counts = clean(raw)
    cleaned["split"] = assign_split(cleaned["id"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(processed_dir / "trips_clean.parquet", index=False)
    report = {
        "source": "Kaggle 'NYC Taxi Trip Duration' competition train.csv (NYC TLC yellow cab trips, 2016)",
        "archive": archive.name,
        "raw_rows": int(raw_rows),
        "cleaning": rule_counts,
        "removed_total": int(raw_rows - len(cleaned)),
        "clean_rows": int(len(cleaned)),
        "split_sizes": {r: int((cleaned["split"] == r).sum()) for r in ROLES},
        "split_allocation_per_mille": {name: hi - lo for name, lo, hi in config.SPLIT_BOUNDS},
        "pickup_datetime_min": cleaned["pickup_datetime"].min().isoformat(),
        "pickup_datetime_max": cleaned["pickup_datetime"].max().isoformat(),
        "kaggle_test_csv": "Extracted but unused: it has no trip_duration label, so it cannot evaluate models.",
    }
    (processed_dir / "data_report.json").write_text(json.dumps(report, indent=2))
    return report


# --- Role-scoped access ------------------------------------------------------
@dataclass
class TripStore:
    """Reads the processed table one split role at a time.

    Rows of other roles are never materialised by a role read (parquet predicate pushdown),
    and every role requested is recorded in `accessed` so tests can prove test-set isolation.
    """
    path: Path = config.CLEAN_PARQUET
    accessed: list[str] = field(default_factory=list)

    def exists(self) -> bool:
        return Path(self.path).exists()

    def role(self, name: str) -> pd.DataFrame:
        if name not in ROLES:
            raise ValueError(f"Unknown split role {name!r}")
        self.accessed.append(name)
        table = pq.read_table(self.path, filters=[("split", "=", name)])
        df = table.to_pandas()
        # Stable row order independent of file layout.
        return df.sort_values("id", kind="stable").reset_index(drop=True)

    def fit_sample(self, sample_size: int | None, seed: int) -> pd.DataFrame:
        pool = self.role("fit")
        return sample_rows(pool, sample_size, seed)

    def report(self) -> dict | None:
        p = Path(self.path).with_name("data_report.json")
        return json.loads(p.read_text()) if p.exists() else None


def sample_rows(df: pd.DataFrame, sample_size: int | None, seed: int) -> pd.DataFrame:
    """Seeded, content-blind subsample (rows assumed sorted by id)."""
    if sample_size is None or sample_size >= len(df):
        return df
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(df), size=int(sample_size), replace=False))
    return df.iloc[idx].reset_index(drop=True)
