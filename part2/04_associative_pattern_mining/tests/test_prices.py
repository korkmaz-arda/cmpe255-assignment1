"""Illustrative prices must stay out of the science."""

from __future__ import annotations

import json

import numpy as np
import pytest

from basket import config, data, engine, mining, pipeline, recommend, rules


def test_price_seed_does_not_move_any_analytical_output(prepared, monkeypatch):
    """Change the prices, and the mining, rules, graph and ranking must not budge.

    Prices are a display attribute bolted onto a dataset that has none. If they
    could reach the analysis, the analysis would be partly synthetic.
    """
    kwargs = dict(min_support=0.05, min_confidence=0.3, min_lift=1.0,
                  max_len=4, include_reference=False)
    pipeline.run(**kwargs)
    before = {
        "rules": [
            {k: v for k, v in r.items() if k != "price"}
            for r in json.loads(config.RULES_JSON.read_text())["rules"]
        ],
        "graph": json.loads(config.GRAPH_JSON.read_text()),
    }
    before_order = [s.product_id for s in engine.load().recommend([1, 2]).suggestions]

    # Re-prepare with a different price seed, then re-mine.
    monkeypatch.setattr(config, "PRICE_SEED", config.PRICE_SEED + 9999)
    data.prepare(n_orders=105, archive=config.RAW_ARCHIVE)
    pipeline.run(**kwargs)

    after = {
        "rules": [
            {k: v for k, v in r.items() if k != "price"}
            for r in json.loads(config.RULES_JSON.read_text())["rules"]
        ],
        "graph": json.loads(config.GRAPH_JSON.read_text()),
    }
    after_engine = engine.load()

    assert after["rules"] == before["rules"]
    assert after["graph"] == before["graph"]
    assert [s.product_id for s in after_engine.recommend([1, 2]).suggestions] == before_order

    # ... and the prices really did change, or the test proves nothing.
    prices = {p["product_id"]: p["price_illustrative"]
              for p in json.loads(config.CATALOG_JSON.read_text())["products"]}
    assert prices


def test_graph_artifact_has_no_price_field(prepared):
    pipeline.run(min_support=0.05, min_confidence=0.3, min_lift=1.0, max_len=4,
                 include_reference=False)
    payload = json.loads(config.GRAPH_JSON.read_text())
    blob = json.dumps(payload)
    assert "price" not in blob


def test_economics_are_only_ever_sums_of_illustrative_prices(prepared):
    pipeline.run(min_support=0.05, min_confidence=0.3, min_lift=1.0, max_len=4,
                 include_reference=False)
    served = engine.load()
    result = served.recommend([1])
    total = sum(served.catalog[pid]["price_illustrative"] for pid in [1])
    assert result.basket_value_illustrative == pytest.approx(total, abs=0.01)
