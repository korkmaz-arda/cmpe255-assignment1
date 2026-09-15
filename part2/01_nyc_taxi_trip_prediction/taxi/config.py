"""Central constants: paths, domain thresholds, transit hubs, split allocation, fare card.

Everything here is plain data so the analytical core stays framework-independent.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR = PROJECT_ROOT / "logs"

KAGGLE_ARCHIVE = RAW_DIR / "nyc-taxi-trip-duration.zip"
CLEAN_PARQUET = PROCESSED_DIR / "trips_clean.parquet"
DATA_REPORT = PROCESSED_DIR / "data_report.json"

# --- Columns -----------------------------------------------------------------
# Fields known at ride-request time. These are the ONLY columns build_features may read.
REQUEST_COLUMNS = [
    "pickup_datetime",
    "passenger_count",
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
# Post-trip / identity fields. Used for the target, cleaning and splitting only - never predictors.
TARGET_COLUMN = "trip_duration"
POST_TRIP_COLUMNS = ["dropoff_datetime", "trip_duration"]
ID_COLUMN = "id"

# --- Domain thresholds -------------------------------------------------------
MIN_DURATION_S = 60           # one minute
MAX_DURATION_S = 3 * 3600     # three hours
MAX_IMPLIED_SPEED_KMH = 120.0  # great-circle speed above this is a GPS/clock fault
MIN_PASSENGERS, MAX_PASSENGERS = 1, 6
# NYC bounding box, wide enough to include Newark Liberty (EWR).
LAT_MIN, LAT_MAX = 40.49, 40.92
LON_MIN, LON_MAX = -74.27, -73.68

RUSH_HOURS = frozenset({7, 8, 9, 16, 17, 18, 19})  # weekday only
LATE_NIGHT_HOURS = frozenset({0, 1, 2, 3, 4, 5})

# Transit hubs used for proximity features: (key, display name, lat, lon)
HUBS = [
    ("jfk", "JFK Airport", 40.6413, -73.7781),
    ("lga", "LaGuardia Airport", 40.7769, -73.8740),
    ("ewr", "Newark Airport", 40.6895, -74.1745),
    ("times_sq", "Times Square", 40.7580, -73.9855),
    ("wall_st", "Wall Street", 40.7060, -74.0086),
    ("grand_central", "Grand Central", 40.7527, -73.9772),
]
AIRPORT_RADIUS_KM = 2.0  # "endpoint at the airport" radius (fare fee and airport error tier)

# --- Split allocation by stable id hash (per-mille buckets) ------------------
SPLIT_SALT = "nyc-taxi-trip-duration/split-v1"
SPLIT_BOUNDS = [  # (role, lower inclusive, upper exclusive) in [0, 1000)
    ("fit", 0, 650),
    ("calibration", 650, 700),
    ("validation", 700, 850),
    ("test", 850, 1000),
]

# --- Modelling defaults ------------------------------------------------------
DEFAULT_XGB = {
    "n_estimators": 250,
    "max_depth": 7,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
}
EARLY_STOPPING_ROUNDS = 30
RETRAIN_BOUNDS = {
    "sample_size": (10_000, None),  # upper bound = size of the fit pool
    "n_estimators": (50, 400),
    "max_depth": (4, 10),
    "learning_rate": (0.01, 0.30),
    "subsample": (0.5, 1.0),
}
BAND_LEVEL = 0.90
BAND_MIN_TIER_ROWS = 200
MIN_PREDICTED_DURATION_S = 60.0
AUTORESEARCH_EPSILON = 1e-4

# --- Illustrative fare card --------------------------------------------------
# Rule-based illustrative estimate; NOT a TLC meter quote and not fitted to any fare data.
FARE_CARD = {
    "base_fare": 3.00,
    "per_mile": 3.50,           # applied to city-block distance
    "per_minute": 0.70,         # applied to the *predicted* duration
    "rush_hour_surcharge": 2.50,  # weekday rush hours only
    "late_night_surcharge": 1.00,
    "congestion_fee": 2.50,     # always applied
    "airport_fee": 1.75,        # either endpoint within AIRPORT_RADIUS_KM of JFK or LGA
}
KM_PER_MILE = 1.609344
