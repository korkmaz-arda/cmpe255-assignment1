"""Download the four public datasets and normalise them into a local parquet cache.

Network access happens here and nowhere else. Every other module reads parquet from
``data/cache`` and raises :class:`DataNotPreparedError` if it is missing, so the app can
show an honest "run the prepare step" message instead of inventing numbers.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from skills_lab.config import (
    CACHE_DIR,
    CACHE_FILES,
    ONLINE_RETAIL_II_URL,
    OPENML_AMES,
    OPENML_CREDITCARD,
    OPENML_TITANIC,
    RAW_DIR,
)


class DataNotPreparedError(RuntimeError):
    """Raised when a required parquet cache file has not been created yet."""

    def __init__(self, key: str) -> None:
        super().__init__(
            f"Dataset '{key}' is not prepared. "
            f"Run:  python scripts/prepare_data.py  (expected at {CACHE_FILES[key]})"
        )
        self.key = key


def cache_path(key: str) -> Path:
    return CACHE_FILES[key]


def is_prepared(key: str) -> bool:
    return CACHE_FILES[key].exists()


def missing_datasets() -> list[str]:
    return [key for key in CACHE_FILES if not is_prepared(key)]


def load_cached(key: str) -> pd.DataFrame:
    """Read a prepared dataset, or explain how to prepare it."""
    path = CACHE_FILES[key]
    if not path.exists():
        raise DataNotPreparedError(key)
    return pd.read_parquet(path)


# --- downloaders ---------------------------------------------------------------------


def _fetch_openml(data_id: int) -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(
        data_id=data_id, as_frame=True, data_home=str(RAW_DIR), parser="auto"
    )
    frame = bunch.frame.copy()
    frame.columns = [str(c) for c in frame.columns]
    return frame


def _download_retail() -> pd.DataFrame:
    """Online Retail II ships as a two-sheet xlsx inside a zip on the UCI archive."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    local_zip = RAW_DIR / "online_retail_ii.zip"
    if not local_zip.exists():
        with urllib.request.urlopen(ONLINE_RETAIL_II_URL, timeout=300) as response:
            local_zip.write_bytes(response.read())

    with zipfile.ZipFile(local_zip) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith(".xlsx"))
        payload = archive.read(name)

    sheets = pd.read_excel(io.BytesIO(payload), sheet_name=None, engine="openpyxl")
    frame = pd.concat(sheets.values(), ignore_index=True)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def prepare(key: str, *, force: bool = False) -> pd.DataFrame:
    """Download (if needed) and cache one dataset, returning the cached frame."""
    path = CACHE_FILES[key]
    if path.exists() and not force:
        return pd.read_parquet(path)

    if key == "titanic":
        frame = _fetch_openml(OPENML_TITANIC)
    elif key == "ames":
        frame = _fetch_openml(OPENML_AMES)
    elif key == "creditcard":
        frame = _fetch_openml(OPENML_CREDITCARD)
    elif key == "retail":
        frame = _download_retail()
    else:  # pragma: no cover - guarded by CACHE_FILES keys
        raise KeyError(f"Unknown dataset key: {key}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Object columns holding mixed types confuse parquet; normalise to string.
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("string")
    frame.to_parquet(path, index=False)
    return frame


def prepare_all(*, force: bool = False) -> dict[str, tuple[int, int]]:
    """Prepare every dataset; returns {key: (rows, columns)}."""
    shapes: dict[str, tuple[int, int]] = {}
    for key in CACHE_FILES:
        frame = prepare(key, force=force)
        shapes[key] = frame.shape
    return shapes
