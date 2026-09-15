"""Versioned, immutable model artifacts with an atomically replaced `current.json` pointer (F05)."""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config, landmarks

VERSION_FILES = ("model.ubj", "metadata.json", "experiments.json", "diagnostics.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_version_id() -> str:
    return datetime.now(timezone.utc).strftime("v%Y%m%dT%H%M%S%fZ")


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


@dataclass
class ModelBundle:
    version: str
    path: Path
    model: xgb.XGBRegressor
    metadata: dict
    experiments: list
    diagnostics: dict
    holdout: dict | None

    @property
    def features(self) -> list[str]:
        return self.metadata["features"]

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        best = self.metadata.get("best_iteration")
        rng = (0, best + 1) if best is not None else None
        return self.model.predict(X[self.features], iteration_range=rng)


class ArtifactStore:
    def __init__(self, root: Path = config.ARTIFACTS_DIR):
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.pointer = self.root / "current.json"
        self.holdout_log = self.root / "holdout_log.json"

    # -- writing ---------------------------------------------------------------
    def staging_dir(self, version: str) -> Path:
        d = self.versions_dir / version
        if d.exists():
            raise FileExistsError(f"Version directory already exists: {d}")
        d.mkdir(parents=True)
        return d

    def discard(self, version: str) -> None:
        shutil.rmtree(self.versions_dir / version, ignore_errors=True)

    def validate_version(self, version: str) -> ModelBundle:
        """Full check before promotion: files, JSON, model reload, finite predictions on probes."""
        d = self.versions_dir / version
        missing = [f for f in VERSION_FILES if not (d / f).exists()]
        if missing:
            raise RuntimeError(f"Version {version} is incomplete; missing {missing}")
        bundle = self._load_dir(version)
        if len(bundle.features) != 20:
            raise RuntimeError(f"Expected 20 features, found {len(bundle.features)}")
        from .features import build_features  # local import to avoid a cycle
        probe = probe_trips()
        pred = bundle.predict_log(build_features(probe))
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            raise RuntimeError("Model produced non-finite or non-positive log predictions on probe trips")
        return bundle

    def promote(self, version: str) -> None:
        """Atomically point `current.json` at an already validated version."""
        write_json_atomic(self.pointer, {"version": version, "promoted_at": utc_now_iso()})

    # -- reading ---------------------------------------------------------------
    def current_version(self) -> str | None:
        if not self.pointer.exists():
            return None
        try:
            return json.loads(self.pointer.read_text())["version"]
        except (json.JSONDecodeError, KeyError):
            return None

    def load_current(self) -> ModelBundle | None:
        version = self.current_version()
        return self._load_dir(version) if version else None

    def load_version(self, version: str) -> ModelBundle:
        return self._load_dir(version)

    def _load_dir(self, version: str) -> ModelBundle:
        d = self.versions_dir / version
        model = xgb.XGBRegressor()
        model.load_model(d / "model.ubj")
        read = lambda name: json.loads((d / name).read_text())  # noqa: E731
        holdout = read("holdout.json") if (d / "holdout.json").exists() else None
        return ModelBundle(version=version, path=d, model=model, metadata=read("metadata.json"),
                           experiments=read("experiments.json"), diagnostics=read("diagnostics.json"),
                           holdout=holdout)

    def list_versions(self) -> list[str]:
        if not self.versions_dir.exists():
            return []
        return sorted(p.name for p in self.versions_dir.iterdir() if (p / "metadata.json").exists())

    def read_holdout_log(self) -> list[dict]:
        return json.loads(self.holdout_log.read_text()) if self.holdout_log.exists() else []

    # -- AutoResearch sessions --------------------------------------------------
    @property
    def sessions_dir(self) -> Path:
        return self.root / "autoresearch"

    def save_session(self, session: dict) -> Path:
        path = self.sessions_dir / f"{session['session_id']}.json"
        write_json_atomic(path, session)
        write_json_atomic(self.sessions_dir / "latest.json", {"session_id": session["session_id"]})
        return path

    def latest_session_id(self) -> str | None:
        ptr = self.sessions_dir / "latest.json"
        if not ptr.exists():
            return None
        try:
            return json.loads(ptr.read_text())["session_id"]
        except (json.JSONDecodeError, KeyError):
            return None

    def load_session(self, session_id: str) -> dict | None:
        path = self.sessions_dir / f"{session_id}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def load_latest_session(self) -> dict | None:
        sid = self.latest_session_id()
        return self.load_session(sid) if sid else None


def probe_trips() -> pd.DataFrame:
    """Fixed landmark-to-landmark trips used to sanity-check a model before promotion."""
    lms = landmarks.LANDMARKS
    rows = []
    for i, a in enumerate(lms):
        b = lms[(i + 3) % len(lms)]
        rows.append({"pickup_datetime": pd.Timestamp("2016-03-15 08:30:00"), "passenger_count": 1,
                     "pickup_latitude": a.lat, "pickup_longitude": a.lon,
                     "dropoff_latitude": b.lat, "dropoff_longitude": b.lon})
    return pd.DataFrame(rows)
