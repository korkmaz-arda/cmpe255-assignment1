"""Application-layer state: caching, eager preload, and honest failure reporting.

All Streamlit-specific concerns live in this layer. ``skills_lab`` knows nothing about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import streamlit as st

from skills_lab.benchmarks import analytics, classification, imbalanced, quality, regression
from skills_lab.catalog import View
from skills_lab.data.acquire import DataNotPreparedError, missing_datasets
from skills_lab.status import status

BENCHMARK_RUNNERS = {
    View.CLASSIFICATION.value: classification.run,
    View.REGRESSION.value: regression.run,
    View.FRAUD.value: imbalanced.run,
    View.ANALYTICS.value: analytics.run,
    View.QUALITY.value: quality.run,
}


@dataclass
class LabState:
    results: dict[str, object] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    missing_data: list[str] = field(default_factory=list)

    def result(self, view: View | str):
        key = view.value if isinstance(view, View) else view
        return self.results.get(key)

    def error(self, view: View | str) -> str | None:
        key = view.value if isinstance(view, View) else view
        return self.errors.get(key)


@st.cache_resource(show_spinner=False)
def load_lab() -> LabState:
    """Compute every benchmark once and keep it for the process lifetime (S05).

    A failure is recorded per workspace rather than swallowed: the affected view shows
    what went wrong instead of stand-in numbers.
    """
    state = LabState(missing_data=missing_datasets())
    for key, runner in BENCHMARK_RUNNERS.items():
        try:
            state.results[key] = runner()
        except DataNotPreparedError as exc:
            state.errors[key] = str(exc)
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the UI
            state.errors[key] = f"{type(exc).__name__}: {exc}"
    return state


def service_status() -> dict:
    return status()


def reset() -> None:
    load_lab.clear()
