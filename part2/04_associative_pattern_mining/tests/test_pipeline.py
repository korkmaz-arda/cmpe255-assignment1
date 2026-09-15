"""Pipeline, persistence, and the controls that must actually do something."""

from __future__ import annotations

import json

import pytest

from basket import config, data, engine, pipeline
from tests.conftest import N_ORDERS


@pytest.fixture
def mined(prepared):
    return pipeline.run(min_support=0.05, min_confidence=0.3, min_lift=1.0, max_len=4)


def test_pipeline_writes_every_artifact(mined):
    for path in (config.RULES_JSON, config.GRAPH_JSON, config.BENCHMARKS_JSON,
                 config.CATALOG_JSON, config.RUN_META_JSON):
        assert path.exists(), path.name
    assert mined["headline"]["n_transactions"] == N_ORDERS


def test_benchmark_compares_like_with_like(mined):
    summary = json.loads(config.BENCHMARKS_JSON.read_text())
    assert summary["implementations_agree"]["agrees"], summary["implementations_agree"]

    implemented = [r for r in summary["leaderboard"] if not r["is_reference"]]
    assert len(implemented) == 3
    # All three mined the same corpus at the same thresholds, so they agree on
    # everything except how long they took.
    assert len({r["n_itemsets"] for r in implemented}) == 1
    assert len({r["n_rules"] for r in implemented}) == 1
    assert len({round(r["elapsed_seconds"], 9) for r in implemented}) == 3


def test_champion_is_the_fastest_implementation_not_the_reference(mined):
    summary = json.loads(config.BENCHMARKS_JSON.read_text())
    implemented = [r for r in summary["leaderboard"] if not r["is_reference"]]
    times = [r["elapsed_seconds"] for r in implemented]
    assert times == sorted(times)
    assert implemented[0]["is_champion"]
    for row in summary["leaderboard"]:
        if row["is_reference"]:
            assert not row["is_champion"]
            assert row["kind"] == "External Reference"
            assert "comparability_note" in row


def test_reference_row_confirms_the_implementations(mined):
    summary = json.loads(config.BENCHMARKS_JSON.read_text())
    row = next(r for r in summary["leaderboard"] if r["is_reference"])
    assert row["agrees_with_implementations"]


def test_presets_are_derived_from_mined_rules(mined):
    payload = json.loads(config.RULES_JSON.read_text())
    assert payload["presets"]
    antecedents = {tuple(r["antecedent"]) for r in payload["rules"]}
    for preset in payload["presets"]:
        assert tuple(preset["items"]) in antecedents


def test_catalog_artifact_labels_the_synthetic_prices(mined):
    payload = json.loads(config.CATALOG_JSON.read_text())
    assert "synthetic" in payload["price_note"].lower()
    assert all("price_illustrative" in p for p in payload["products"])


@pytest.mark.parametrize(
    "override",
    [
        {"min_support": 0.12},
        {"min_confidence": 0.95},
        {"min_lift": 3.0},
        {"n_orders": 60},
    ],
)
def test_every_remining_control_changes_the_result(prepared, override):
    """The source shipped three sliders that did nothing. Each one must bite."""
    base_kwargs = dict(min_support=0.05, min_confidence=0.3, min_lift=1.0,
                       max_len=4, include_reference=False)
    baseline = pipeline.run(**base_kwargs)["headline"]
    changed = pipeline.run(**{**base_kwargs, **override})["headline"]

    if "n_orders" in override:
        assert changed["n_transactions"] == 60
        assert changed["n_transactions"] != baseline["n_transactions"]
    else:
        assert changed["n_active_rules"] != baseline["n_active_rules"], override


def test_engine_serves_from_artifacts_and_reloads(mined):
    loaded = engine.load()
    assert loaded.health()["ready"]
    assert loaded.n_rules == mined["headline"]["n_active_rules"]

    stamp = engine.version_stamp()
    pipeline.run(min_support=0.05, min_confidence=0.9, min_lift=1.0, max_len=4,
                 include_reference=False)
    assert engine.version_stamp() != stamp
    assert engine.load().n_rules != loaded.n_rules


def test_engine_reports_missing_artifacts_clearly():
    with pytest.raises(engine.ArtifactsMissing, match="basket.pipeline"):
        engine.load()


def test_basket_coverage_is_reported(mined):
    coverage = mined["headline"]["basket_coverage"]
    assert 0.0 <= coverage <= 1.0
    # The fixture's rules fire on the pair blocks but not on the singleton block.
    assert coverage < 1.0
