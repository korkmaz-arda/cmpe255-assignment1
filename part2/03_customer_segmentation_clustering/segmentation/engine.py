"""Artifact-backed serving layer.

Every view is served from on-disk artifacts held in a single long-lived object.
Retraining rewrites those artifacts and the engine is reloaded in place, which is
the mechanism by which a retrain immediately changes what everyone sees.

This module is deliberately plain Python with no UI-framework dependency: the
caching that makes it a shared singleton lives in the app layer, so the whole
serving path stays importable and testable on its own.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import autoresearch, config, inference, model, projections


def _read_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


ARTIFACT_ATTRIBUTES = [
    "MODEL_BUNDLE",
    "PCA_BUNDLE",
    "PERSONAS_JSON",
    "BENCHMARKS_JSON",
    "ELBOW_JSON",
    "SCATTER_JSON",
    "RUN_META_JSON",
    "AUTORESEARCH_JSON",
]


def artifact_paths() -> list:
    """Resolve the artifact paths from config on every call.

    Deliberately not a module-level constant: snapshotting the paths at import
    time would let the version stamp go stale whenever the configured artifact
    directory changes.
    """
    return [getattr(config, name) for name in ARTIFACT_ATTRIBUTES]


def version_stamp() -> str:
    """A cheap fingerprint of the artifact set, used to invalidate caches."""
    return "|".join(
        f"{path.name}:{path.stat().st_mtime_ns if path.exists() else 0}"
        for path in artifact_paths()
    )


@dataclass
class Engine:
    """All served state, loaded once from disk."""

    bundle: model.ModelBundle | None = None
    pca: object | None = None
    personas: list[dict] = field(default_factory=list)
    benchmarks: dict = field(default_factory=dict)
    elbow: dict = field(default_factory=dict)
    scatter: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    research: dict | None = None
    stamp: str = ""

    # -- lifecycle --------------------------------------------------------- #

    @property
    def trained(self) -> bool:
        return self.bundle is not None and bool(self.personas)

    @property
    def has_research(self) -> bool:
        return bool(self.research)

    def reload(self) -> "Engine":
        """Re-read every artifact from disk. Missing artifacts degrade to empty."""
        try:
            self.bundle = model.load()
            self.pca = projections.load_pca()
        except FileNotFoundError:
            self.bundle, self.pca = None, None

        self.personas = _read_json(config.PERSONAS_JSON, []) or []
        self.benchmarks = _read_json(config.BENCHMARKS_JSON, {}) or {}
        self.elbow = _read_json(config.ELBOW_JSON, {}) or {}
        self.scatter = _read_json(config.SCATTER_JSON, {}) or {}
        self.meta = _read_json(config.RUN_META_JSON, {}) or {}
        self.research = autoresearch.load()
        self.stamp = version_stamp()
        return self

    # -- accessors --------------------------------------------------------- #

    @property
    def persona_by_cluster(self) -> dict[int, dict]:
        return {p["cluster"]: p for p in self.personas}

    @property
    def production(self) -> dict:
        return self.benchmarks.get("production", {})

    @property
    def leaderboard(self) -> list[dict]:
        return self.benchmarks.get("leaderboard", [])

    def classify(self, payload: dict) -> dict:
        """Classify one customer against the served model (F09)."""
        if not self.trained:
            raise RuntimeError("No trained model is loaded. Run the pipeline first.")
        return inference.classify(payload, self.bundle, self.pca, self.persona_by_cluster)


def load() -> Engine:
    return Engine().reload()
