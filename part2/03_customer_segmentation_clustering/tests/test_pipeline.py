"""Pipeline orchestration, artifact persistence and the serving engine."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from segmentation import config, engine as engine_module, model, pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def trained(clean_sample):
    return pipeline.run(k=4, seed=3, frame=clean_sample)


def test_pipeline_writes_every_artifact(trained):
    for path in engine_module.artifact_paths():
        if path is config.AUTORESEARCH_JSON:
            continue  # written by the AutoResearch entry point, not the pipeline
        assert path.exists(), f"missing artifact: {path.name}"


def test_model_bundle_round_trips(trained):
    bundle = model.load()
    assert bundle.k == 4
    assert bundle.columns == list(config.FEATURE_COLUMNS)
    assert bundle.centroids.shape == (4, len(config.FEATURE_COLUMNS))
    # The scaler travels with the model, fitted, not a fresh one.
    assert hasattr(bundle.scaler, "mean_")


def test_cluster_count_actually_reaches_the_pipeline(clean_sample):
    """The source validated this control and then ignored it."""
    for k in (3, 6):
        result = pipeline.run(k=k, seed=3, frame=clean_sample)
        assert result["benchmarks"]["k"] == k
        assert model.load().k == k
        assert len(result["personas"]) == k
        assert result["benchmarks"]["production"]["n_clusters"] == k


def test_out_of_range_cluster_count_is_rejected(clean_sample):
    with pytest.raises(ValueError, match="between"):
        pipeline.run(k=config.MAX_K + 1, seed=3, frame=clean_sample)
    with pytest.raises(ValueError, match="between"):
        pipeline.run(k=1, seed=3, frame=clean_sample)


def test_training_sample_size_takes_effect(clean_sample, monkeypatch):
    monkeypatch.setattr(config, "MIN_TRAIN_SAMPLE", 50)
    full = pipeline.run(k=3, seed=3, frame=clean_sample)
    reduced = pipeline.run(k=3, seed=3, n_rows=120, frame=clean_sample)
    assert full["meta"]["n_customers"] == len(clean_sample)
    assert reduced["meta"]["n_customers"] == 120
    assert reduced["meta"]["trained_on_full_dataset"] is False


def test_undersized_training_sample_is_rejected(clean_sample):
    with pytest.raises(ValueError, match="at least"):
        pipeline.run(k=3, seed=3, n_rows=10, frame=clean_sample)


def test_pipeline_is_deterministic(clean_sample):
    first = pipeline.run(k=4, seed=11, frame=clean_sample)
    second = pipeline.run(k=4, seed=11, frame=clean_sample)
    assert (first["benchmarks"]["production"]["silhouette"]
            == pytest.approx(second["benchmarks"]["production"]["silhouette"]))
    assert [p["key"] for p in first["personas"]] == [p["key"] for p in second["personas"]]


def test_production_and_leaderboard_share_the_evaluation_matrix(trained):
    """K-Means appears in both; fit on the same matrix with the same k it must
    reach the same silhouette, which is what makes the leaderboard meaningful."""
    benchmarks = trained["benchmarks"]
    kmeans_row = next(r for r in benchmarks["leaderboard"] if r["algorithm"] == "K-Means")
    production = benchmarks["production"]
    assert kmeans_row["silhouette"] == pytest.approx(production["silhouette"], abs=1e-6)


def test_scatter_sample_carries_both_projections(trained):
    scatter = json.loads(config.SCATTER_JSON.read_text())
    assert scatter["points"]
    for point in scatter["points"][:20]:
        assert np.isfinite([point["pca_x"], point["pca_y"], point["tsne_x"], point["tsne_y"]]).all()
        assert point["cluster"] in range(4)
        for attribute in config.BASE_FEATURES:
            assert attribute in point


def test_meta_reports_the_real_dataset(trained):
    meta = trained["meta"]
    assert meta["dataset"]["synthetic"] is False
    assert meta["dataset"]["name"] == config.DATASET_NAME
    assert 0.0 < meta["pca_explained_variance"]["total"] <= 1.0


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

def test_engine_loads_artifacts_and_classifies(trained):
    engine = engine_module.load()
    assert engine.trained
    assert len(engine.personas) == 4
    result = engine.classify(dict(config.INPUT_DEFAULTS))
    assert result["cluster"] in range(4)
    assert result["persona"]["name"]


def test_engine_degrades_gracefully_without_artifacts():
    engine = engine_module.Engine().reload()
    assert engine.trained is False
    assert engine.personas == []
    assert engine.leaderboard == []
    assert engine.production == {}
    with pytest.raises(RuntimeError, match="No trained model"):
        engine.classify(dict(config.INPUT_DEFAULTS))


def test_version_stamp_changes_when_artifacts_are_rewritten(clean_sample):
    pipeline.run(k=3, seed=1, frame=clean_sample)
    before = engine_module.version_stamp()
    pipeline.run(k=5, seed=1, frame=clean_sample)
    assert engine_module.version_stamp() != before


def test_retrain_changes_what_the_engine_serves(clean_sample):
    """Hot reload: rewriting artifacts must change every served view."""
    pipeline.run(k=3, seed=1, frame=clean_sample)
    before = engine_module.load()
    assert len(before.personas) == 3

    pipeline.run(k=6, seed=1, frame=clean_sample)
    after = engine_module.load()
    assert len(after.personas) == 6
    assert after.production["n_clusters"] == 6
    assert after.meta["k"] == 6


# --------------------------------------------------------------------------- #
# Framework independence
# --------------------------------------------------------------------------- #

def test_segmentation_package_imports_without_streamlit():
    """The data-science core must never depend on the UI framework."""
    script = """
import importlib, sys

class Blocker:
    def find_module(self, name, path=None):
        return self.find_spec(name, path)
    def find_spec(self, name, path=None, target=None):
        if name == 'streamlit' or name.startswith('streamlit.'):
            raise ImportError('streamlit is deliberately unavailable in this test')
        return None

sys.meta_path.insert(0, Blocker())
for module in ['config', 'data', 'features', 'tournament', 'model', 'projections',
               'personas', 'elbow', 'inference', 'autoresearch', 'pipeline', 'engine']:
    importlib.import_module('segmentation.' + module)
assert 'streamlit' not in sys.modules
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_no_streamlit_import_in_the_core_package():
    for path in (PROJECT_ROOT / "segmentation").glob("*.py"):
        source = path.read_text()
        assert "import streamlit" not in source, f"{path.name} imports streamlit"
