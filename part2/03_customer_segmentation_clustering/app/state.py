"""Shared application state and the only Streamlit caching in the project.

The data-science package is framework-independent; this module is the single
place that knows about Streamlit's cache. It wraps the plain-Python
``segmentation.engine.Engine`` in a cached resource keyed on the artifact
version stamp, so a retrain — which rewrites the artifacts — is picked up by
every session without a restart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import autoresearch, config, engine as engine_module, pipeline  # noqa: E402


@st.cache_resource(show_spinner=False)
def _load_engine(stamp: str) -> engine_module.Engine:
    """Load artifacts once per artifact version. ``stamp`` is the cache key."""
    return engine_module.load()


def get_engine() -> engine_module.Engine:
    """The shared engine, reloaded automatically when artifacts change on disk."""
    return _load_engine(engine_module.version_stamp())


def invalidate() -> None:
    """Drop the cached engine so the next read re-reads the new artifacts."""
    _load_engine.clear()


def retrain(k: int, seed: int, n_rows: int | None) -> dict:
    """Re-run the whole pipeline and hot-reload the serving layer (F11)."""
    result = pipeline.run(k=k, seed=seed, n_rows=n_rows)
    invalidate()
    return result


def run_autoresearch(n_rows: int, k: int, seed: int) -> dict:
    """Run a fresh search, replacing the stored run record (F12)."""
    record = autoresearch.execute(n_rows=n_rows, k=k, seed=seed)
    autoresearch.save(record)
    invalidate()
    return record


VIEW_EXPLORER = "Segment explorer"
VIEW_ADMIN = "Admin console"
VIEWS = [VIEW_EXPLORER, VIEW_ADMIN]
ADMIN_TABS = ["Benchmarks", "AutoResearch", "Radar profiles"]
PROJECTIONS = ["PCA", "t-SNE"]


def current_view() -> str:
    return st.session_state.get("view_control") or VIEW_EXPLORER


def current_admin_tab() -> str:
    return st.session_state.get("admin_tab_control") or ADMIN_TABS[0]


def current_projection() -> str:
    return st.session_state.get("projection_control") or PROJECTIONS[0]


def init_session() -> None:
    """Ephemeral client state: view, sub-tab, filters, selections."""
    # Controls are keyed by their widget key and read back through the helpers
    # below. A widget key IS its session-state slot, so keeping a second copy
    # under a different name gives two competing sources of truth: the widget
    # silently wins on every rerun, and anything that sets the copy is ignored.
    defaults = {
        "view_control": VIEW_EXPLORER,
        "admin_tab_control": ADMIN_TABS[0],
        "selected_cluster": None,
        "projection_control": "PCA",
        "prediction": None,
        "prediction_error": None,
        "phase_filter": "All phases",
        "decision_filter": "All decisions",
        "radar_focus": "All clusters",
        "inspect_step": None,
        "action_message": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def persona_color(engine, cluster: int) -> str:
    persona = engine.persona_by_cluster.get(cluster)
    return persona["color"] if persona else config.FALLBACK_COLOR


def persona_name(engine, cluster: int) -> str:
    persona = engine.persona_by_cluster.get(cluster)
    return persona["name"] if persona else f"Cluster {cluster}"
