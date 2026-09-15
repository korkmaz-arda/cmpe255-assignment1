"""Shared fixtures. Every test runs offline against a small committed sample."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import data  # noqa: E402

FIXTURE_CSV = Path(__file__).parent / "fixtures" / "marketing_campaign_sample.csv"


@pytest.fixture(scope="session")
def raw_sample() -> pd.DataFrame:
    """The raw-schema fixture, including the deliberate edge-case rows."""
    return pd.read_csv(FIXTURE_CSV, sep="\t")


@pytest.fixture(scope="session")
def clean_sample(raw_sample: pd.DataFrame) -> pd.DataFrame:
    return data.prepare(raw_sample)


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    """Point every artifact path at a temp directory so tests never touch real ones."""
    from segmentation import config

    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    for name in ("MODEL_BUNDLE", "PCA_BUNDLE"):
        monkeypatch.setattr(config, name, tmp_path / f"{name.lower()}.joblib")
    for name in ("PERSONAS_JSON", "BENCHMARKS_JSON", "ELBOW_JSON",
                 "SCATTER_JSON", "AUTORESEARCH_JSON", "RUN_META_JSON"):
        monkeypatch.setattr(config, name, tmp_path / f"{name.lower()}.json")
    yield
