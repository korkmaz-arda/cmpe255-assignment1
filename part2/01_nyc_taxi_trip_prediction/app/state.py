"""Streamlit-specific caching around the framework-independent core."""
from __future__ import annotations

import streamlit as st

from taxi import config
from taxi.artifacts import ArtifactStore, ModelBundle
from taxi.data import TripStore


def artifact_store() -> ArtifactStore:
    return ArtifactStore(config.ARTIFACTS_DIR)


def trip_store() -> TripStore:
    return TripStore(config.CLEAN_PARQUET)


@st.cache_resource(show_spinner="Loading model version…")
def _load_bundle(version: str) -> ModelBundle:
    store = artifact_store()
    return store.load_version(version)  # cache keyed by immutable version id


def active_bundle() -> tuple[ModelBundle | None, str | None]:
    """Return (bundle, error). Cache key is the version named by current.json."""
    store = artifact_store()
    version = store.current_version()
    if version is None:
        return None, None
    try:
        bundle = _load_bundle(version)
    except Exception as exc:  # corrupt or missing version directory
        return None, f"Could not load model version {version}: {exc}"
    # holdout.json may be added to a version after it was cached (finalize.py), so read it fresh.
    holdout_path = bundle.path / "holdout.json"
    if (holdout_path.exists()) != (bundle.holdout is not None):
        _load_bundle.clear()
        bundle = _load_bundle(version)
    return bundle, None


@st.cache_data(show_spinner=False)
def _load_session(session_id: str) -> dict | None:
    return artifact_store().load_session(session_id)


def latest_session() -> dict | None:
    sid = artifact_store().latest_session_id()
    return _load_session(sid) if sid else None


def data_report() -> dict | None:
    return trip_store().report()


def clear_caches() -> None:
    _load_bundle.clear()
    _load_session.clear()
