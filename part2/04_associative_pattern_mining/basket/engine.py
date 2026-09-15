"""In-process serving layer: load the artifacts once, answer from memory.

The application never mines at request time. It reads the rule set, the affinity
graph, the benchmark summary, the catalog and the search history that
`pipeline.py` and `autoresearch.py` produced, and serves lookups over them. A
re-mine rewrites those files; `version_stamp` changes, and the application layer
drops its cached engine and reloads.

Plain Python, no UI framework: the whole package stays importable and testable
with Streamlit absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config
from .recommend import Recommendation, recommend
from .rules import Rule, from_record


class ArtifactsMissing(RuntimeError):
    """Raised when the serving layer has nothing to serve."""


def required_artifacts() -> tuple[Path, ...]:
    """Read from config on every call, so a redirected path is honoured."""
    return (
        config.RULES_JSON,
        config.GRAPH_JSON,
        config.BENCHMARKS_JSON,
        config.CATALOG_JSON,
    )


def artifacts_present() -> bool:
    return all(path.exists() for path in required_artifacts())


def version_stamp() -> str:
    """Cheap fingerprint of the artifacts on disk; changes whenever they are rewritten."""
    parts = []
    for path in required_artifacts() + (config.AUTORESEARCH_JSON, config.RUN_META_JSON):
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
        except FileNotFoundError:
            parts.append(f"{path.name}:missing")
    return "|".join(parts)


def _read(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@dataclass
class Engine:
    rules: list[Rule]
    rule_records: list[dict]
    graph: dict
    benchmarks: dict
    catalog: dict[int, dict]
    run_meta: dict
    search: dict | None = None
    stamp: str = ""
    _names: dict[int, str] = field(default_factory=dict)

    # ---------------- catalog ----------------

    @property
    def names(self) -> dict[int, str]:
        if not self._names:
            self._names = {pid: meta["product_name"] for pid, meta in self.catalog.items()}
        return self._names

    def catalog_sorted(self) -> list[dict]:
        return sorted(self.catalog.values(), key=lambda m: m["product_name"].lower())

    # ---------------- serving ----------------

    def recommend(self, basket, limit: int = config.RECOMMENDATION_COUNT) -> Recommendation:
        return recommend(basket, self.rules, self.catalog, limit=limit)

    def top_rules(self, limit: int = config.RULES_TABLE_FEED) -> list[dict]:
        return self.rule_records[:limit]

    @property
    def headline(self) -> dict:
        return self.benchmarks.get("production", {})

    @property
    def champion(self) -> str | None:
        return self.benchmarks.get("champion")

    @property
    def n_rules(self) -> int:
        return len(self.rules)

    @property
    def reference_row(self) -> dict | None:
        for row in self.benchmarks.get("leaderboard", []):
            if row.get("is_reference"):
                return row
        return None

    def health(self) -> dict:
        return {
            "ready": bool(self.rules),
            "n_rules": self.n_rules,
            "champion": self.champion,
            "n_transactions": self.headline.get("n_transactions", 0),
            "catalog_size": len(self.catalog),
            "generated_at": self.run_meta.get("generated_at"),
            "has_search_history": self.search is not None,
        }


def load() -> Engine:
    if not artifacts_present():
        missing = [p.name for p in required_artifacts() if not p.exists()]
        raise ArtifactsMissing(
            "Missing artifacts: " + ", ".join(missing) + ". Run `python -m basket.pipeline`."
        )

    rules_payload = _read(config.RULES_JSON)
    rule_records = rules_payload["rules"]
    catalog_payload = _read(config.CATALOG_JSON)

    search = None
    if config.AUTORESEARCH_JSON.exists():
        search = _read(config.AUTORESEARCH_JSON)

    run_meta = _read(config.RUN_META_JSON) if config.RUN_META_JSON.exists() else {}

    return Engine(
        rules=[from_record(record) for record in rule_records],
        rule_records=rule_records,
        graph=_read(config.GRAPH_JSON),
        benchmarks=_read(config.BENCHMARKS_JSON),
        catalog={int(p["product_id"]): p for p in catalog_payload["products"]},
        run_meta=run_meta,
        search=search,
        stamp=version_stamp(),
    )
