"""Application state: the only place in the project that knows about Streamlit.

The `basket` package is plain Python. This module wraps its artifact engine in a
cached resource keyed on the artifact version stamp, so a re-mine — which
rewrites the artifacts — is picked up by every session without a restart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from basket import autoresearch, config, engine as engine_module, pipeline  # noqa: E402


@st.cache_resource(show_spinner=False)
def _load_engine(stamp: str) -> engine_module.Engine:
    """Load the artifacts once per version. `stamp` is the cache key."""
    return engine_module.load()


def get_engine() -> engine_module.Engine:
    return _load_engine(engine_module.version_stamp())


def invalidate() -> None:
    _load_engine.clear()


def artifacts_ready() -> bool:
    return engine_module.artifacts_present()


def remine(min_support: float, min_confidence: float, min_lift: float, n_orders: int) -> dict:
    """Re-run the pipeline. Every one of these four arguments changes the result."""
    meta = pipeline.run(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        n_orders=n_orders,
    )
    invalidate()
    return meta


def run_search(n_orders: int) -> dict:
    record = autoresearch.execute(n_orders=n_orders)
    autoresearch.save(record)
    invalidate()
    return record


def init_session() -> None:
    """Ephemeral client state: view, tabs, basket, filters, sort, selections."""
    defaults = {
        "view": "Basket workspace",
        "admin_tab": "Algorithm benchmark",
        "basket": None,             # seeded from the artifacts on first load
        "selected_node": None,
        "inspect_step": None,
        "phase_filter": "All phases",
        "decision_filter": "All decisions",
        "rules_query": "",
        "rules_sort": "lift",
        "rules_descending": True,
        "action_message": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def ensure_basket(engine) -> list[int]:
    """Open with a non-empty basket so the workspace shows results immediately."""
    if st.session_state.basket is None:
        presets = default_presets(engine)
        st.session_state.basket = list(presets[0]["items"]) if presets else []
    return st.session_state.basket


def default_presets(engine) -> list[dict]:
    import json
    try:
        with config.RULES_JSON.open(encoding="utf-8") as handle:
            return json.load(handle).get("presets", [])
    except (OSError, ValueError):
        return []


def set_basket(items) -> None:
    st.session_state.basket = list(dict.fromkeys(int(i) for i in items))


def add_to_basket(product_id: int) -> None:
    basket = list(st.session_state.basket or [])
    if int(product_id) not in basket:
        basket.append(int(product_id))
    st.session_state.basket = basket


def remove_from_basket(product_id: int) -> None:
    st.session_state.basket = [i for i in (st.session_state.basket or []) if i != int(product_id)]
