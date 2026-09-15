"""Corpus construction: identity, determinism, and the support denominator."""

from __future__ import annotations

import numpy as np
import pytest

from basket import config, data
from tests.conftest import CATALOG_PRODUCTS, N_ORDERS


def test_catalog_applies_the_frequency_floor(catalog):
    assert len(catalog) == CATALOG_PRODUCTS
    # Product 11 appears in 5 of 105 orders (4.8%), below the 8% floor.
    assert 11 not in set(catalog["product_id"])
    # Product 12 is never ordered at all.
    assert 12 not in set(catalog["product_id"])
    assert (catalog["marginal_frequency"] >= config.CATALOG_MIN_FREQUENCY).all()


def test_catalog_carries_display_metadata_not_keys(catalog):
    row = catalog[catalog["product_id"] == 1].iloc[0]
    assert row["product_name"] == "Alpha Apples"
    assert row["department"] == "produce"
    assert row["color"].startswith("#")
    # The key is the id; names are metadata and are allowed to collide or change.
    assert catalog["product_id"].is_unique


def test_sparse_and_empty_baskets_stay_in_the_denominator(corpus):
    """The invariant the whole metric table rests on.

    Twenty orders keep a single catalog product and five project to nothing at
    all. Dropping them would shrink the denominator and inflate every support.
    """
    assert corpus.n_transactions == N_ORDERS
    assert sum(1 for t in corpus.transactions if len(t) == 0) == 5
    assert sum(1 for t in corpus.transactions if len(t) == 1) == 20
    assert corpus.n_minable == 80


def test_support_denominator_is_the_full_sample(corpus):
    from basket import mining
    result = mining.eclat(corpus, min_support=0.01, max_len=4)
    # Product 1 appears in 65 of the 105 orders.
    assert result.support([1]) == pytest.approx(65 / 105)
    assert result.n_transactions == N_ORDERS


def test_sampling_is_deterministic_and_content_blind(catalog, fixture_archive):
    first = data.build_corpus(catalog, n_orders=50, seed=7, archive=fixture_archive)
    second = data.build_corpus(catalog, n_orders=50, seed=7, archive=fixture_archive)
    other = data.build_corpus(catalog, n_orders=50, seed=8, archive=fixture_archive)
    assert list(first["order_id"]) == list(second["order_id"])
    assert list(first["order_id"]) != list(other["order_id"])
    assert len(first) == 50


def test_illustrative_prices_are_reproducible():
    ids = np.array([5, 1, 9, 3])
    assert list(data.illustrative_prices(ids)) == list(data.illustrative_prices(ids))
    # Keyed on the product id, so a product keeps its price regardless of ordering.
    shuffled = np.array([9, 3, 5, 1])
    by_id = dict(zip(ids, data.illustrative_prices(ids)))
    by_id_shuffled = dict(zip(shuffled, data.illustrative_prices(shuffled)))
    assert by_id == by_id_shuffled


def test_prepare_round_trips_through_parquet(prepared):
    assert prepared["n_orders_sampled"] == N_ORDERS
    assert prepared["catalog_size"] == CATALOG_PRODUCTS
    reloaded = data.load_corpus()
    assert reloaded.n_transactions == N_ORDERS
    assert data.load_catalog().shape[0] == CATALOG_PRODUCTS


def test_missing_data_fails_loudly():
    with pytest.raises(data.DataUnavailable, match="basket.data"):
        data.load_corpus()
