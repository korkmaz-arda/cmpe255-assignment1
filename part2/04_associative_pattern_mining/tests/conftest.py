"""Offline test setup.

Every test runs against a small committed fixture in the real Instacart schema,
packed into a zip archive the same way the live dataset ships. Nothing downloads,
and every path the package writes to is redirected into ``tmp_path``.

The fixture is 105 orders over 11 ordered products, laid out so each property
under test is hand-checkable:

  30 x {1,2}          a plain pair
  15 x {1,2,3}        a triple extending it
  10 x {4,5}          an independent pair
  15 x {6,7,8,9,10}   a genuine 5-itemset, so max_len > 4 has something to find
  20 x {1}            one catalog product: contributes no candidate, still a transaction
   5 x {11}           product below the catalog floor, so the basket projects to EMPTY
  10 x {2,4}          a cross pair

With a catalog floor of 0.08, product 11 is excluded and its five orders become
empty baskets — which must still count toward the support denominator of 105.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from basket import config

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_FILES = ("order_products__train.csv", "products.csv", "aisles.csv", "departments.csv")

N_ORDERS = 105
CATALOG_FLOOR = 0.08          # admits products 1-10, excludes the rare product 11
CATALOG_PRODUCTS = 10


@pytest.fixture(scope="session")
def fixture_archive(tmp_path_factory) -> Path:
    """The fixture CSVs packed as a zip, mirroring how the real dataset ships."""
    archive = tmp_path_factory.mktemp("raw") / "instacart.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FIXTURE_FILES:
            zf.write(FIXTURE_DIR / name, arcname=name)
    return archive


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, fixture_archive, monkeypatch):
    """Redirect every artifact and data path into tmp_path, and lower the catalog floor."""
    data_dir = tmp_path / "data"
    artifact_dir = tmp_path / "artifacts"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "processed").mkdir(parents=True)
    artifact_dir.mkdir(parents=True)

    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    monkeypatch.setattr(config, "RAW_DIR", data_dir / "raw")
    monkeypatch.setattr(config, "PROCESSED_DIR", data_dir / "processed")
    monkeypatch.setattr(config, "ARTIFACT_DIR", artifact_dir)
    monkeypatch.setattr(config, "RAW_ARCHIVE", fixture_archive)
    monkeypatch.setattr(config, "BASKETS_PARQUET", data_dir / "processed" / "baskets.parquet")
    monkeypatch.setattr(config, "CATALOG_PARQUET", data_dir / "processed" / "catalog.parquet")
    monkeypatch.setattr(config, "CORPUS_META_JSON", data_dir / "processed" / "corpus_meta.json")
    for name in ("RULES_JSON", "GRAPH_JSON", "BENCHMARKS_JSON", "CATALOG_JSON",
                 "RUN_META_JSON", "AUTORESEARCH_JSON"):
        monkeypatch.setattr(config, name, artifact_dir / getattr(config, name).name)

    monkeypatch.setattr(config, "CATALOG_MIN_FREQUENCY", CATALOG_FLOOR)
    monkeypatch.setattr(config, "DEFAULT_N_ORDERS", N_ORDERS)
    monkeypatch.setattr(config, "SEARCH_N_ORDERS", N_ORDERS)
    monkeypatch.setattr(config, "TRAIN_SPLIT_ORDERS", N_ORDERS)
    yield


@pytest.fixture
def catalog(fixture_archive):
    from basket import data
    return data.build_catalog(fixture_archive)


@pytest.fixture
def corpus(catalog, fixture_archive):
    from basket import data
    frame = data.build_corpus(catalog, n_orders=N_ORDERS, archive=fixture_archive)
    return data.to_corpus(frame)


@pytest.fixture
def prepared(fixture_archive):
    """Corpus + catalog written to disk, as the pipeline expects to find them."""
    from basket import data
    summary = data.prepare(n_orders=N_ORDERS, archive=fixture_archive)
    return summary
