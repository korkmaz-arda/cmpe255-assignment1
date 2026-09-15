"""The hill climb: gate behaviour, statefulness and a real consensus phase."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from segmentation import autoresearch, config


@pytest.fixture(scope="module")
def frame():
    from pathlib import Path
    from segmentation import data

    raw = pd.read_csv(Path(__file__).parent / "fixtures" / "marketing_campaign_sample.csv", sep="\t")
    return data.prepare(raw)


@pytest.fixture(scope="module")
def record(frame):
    return autoresearch.AutoResearch(frame, k=3, seed=1).run()


def test_search_starts_from_base_attributes_only(record):
    assert record["steps"][0]["active_features"] == list(config.BASE_FEATURES)


def test_all_four_phases_are_present(record):
    phases = {step["phase"] for step in record["steps"]}
    assert phases == {"backbone", "features", "hyperparameters", "consensus"}


def test_every_step_is_fully_inspectable(record):
    for step in record["steps"]:
        for key in ("hypothesis", "transformation", "reflection", "params",
                    "silhouette_before", "silhouette_after", "delta", "decision"):
            assert step[key] not in (None, "", {}), f"step {step['round']} missing {key}"
        assert step["delta"] == pytest.approx(step["silhouette_after"] - step["silhouette_before"])


def test_gate_requires_a_real_improvement():
    search = autoresearch.AutoResearch(pd.DataFrame(columns=config.BASE_FEATURES), k=3)
    healthy = {"smallest_share": 0.3, "sizes": {0: 30}, "n_clusters": 3}
    assert search._consider(0.5 + config.AUTORESEARCH_GATE * 2, 0.5, healthy)[0] is True
    assert search._consider(0.5 + config.AUTORESEARCH_GATE / 2, 0.5, healthy)[0] is False
    assert search._consider(0.4, 0.5, healthy)[0] is False


def test_gate_rejects_degenerate_partitions_however_high_they_score():
    """Silhouette is gameable by quarantining outliers into singleton clusters."""
    search = autoresearch.AutoResearch(pd.DataFrame(columns=config.BASE_FEATURES), k=3)
    gamed = {"smallest_share": 0.0006, "sizes": {0: 1580, 1: 11, 2: 7, 3: 1, 4: 1}, "n_clusters": 5}
    accepted, reason = search._consider(0.95, 0.2, gamed)
    assert accepted is False
    assert reason == "degenerate"


def test_accepted_features_persist_into_later_steps(record):
    """The search is stateful: an accepted feature must stay in the matrix."""
    accepted_features = [
        step["component"] for step in record["steps"]
        if step["phase"] == "features" and step["decision"] == "accepted"
    ]
    for name in accepted_features:
        assert name in record["active_features"]
    rejected = [
        step["component"] for step in record["steps"]
        if step["phase"] == "features" and step["decision"] == "rejected"
    ]
    for name in rejected:
        assert name not in record["active_features"]


def test_rejected_steps_do_not_advance_the_best_score(record):
    best = record["start_silhouette"]
    for step in record["steps"][1:]:
        if step["decision"] == "accepted":
            assert step["silhouette_after"] > best
            best = step["silhouette_after"]
        else:
            assert step["silhouette_before"] == pytest.approx(best)
    assert record["best_silhouette"] == pytest.approx(best)


def test_consensus_phase_computes_a_real_co_assignment_rate(record):
    """The source fabricated this figure. It must be measured from the partitions."""
    step = next(s for s in record["steps"] if s["phase"] == "consensus")
    rate = step["params"]["measured_co_assignment_rate"]
    assert 0.0 <= rate <= 1.0
    # A fabricated constant would not move with the data; a real one is not exactly 0 or 1.
    assert rate not in (0.0, 1.0)
    assert step["silhouette_after"] != step["silhouette_before"] or step["decision"] == "rejected"


def test_consensus_is_scored_not_rubber_stamped(record):
    """The source logged this step as accepted with delta 0 unconditionally."""
    step = next(s for s in record["steps"] if s["phase"] == "consensus")
    if step["decision"] == "accepted":
        assert step["delta"] > config.AUTORESEARCH_GATE
    else:
        assert step["delta"] <= config.AUTORESEARCH_GATE or step["partition"]["smallest_share"] < config.MIN_CLUSTER_SHARE


def test_improvement_is_measured_against_its_own_starting_score(record):
    expected = (record["best_silhouette"] - record["start_silhouette"]) / abs(record["start_silhouette"]) * 100
    assert record["improvement_pct"] == pytest.approx(expected)
    assert "own phase-1 starting score" in record["improvement_basis"]


def test_run_record_does_not_claim_promotion_to_production(record):
    assert record["promoted_to_production"] is False


def test_counts_are_consistent(record):
    accepted = sum(1 for s in record["steps"] if s["decision"] == "accepted")
    rejected = sum(1 for s in record["steps"] if s["decision"] == "rejected")
    assert record["accepted"] == accepted
    assert record["rejected"] == rejected
    assert record["total_steps"] == len(record["steps"])


def test_search_is_deterministic(frame):
    a = autoresearch.AutoResearch(frame, k=3, seed=5).run()
    b = autoresearch.AutoResearch(frame, k=3, seed=5).run()
    assert a["best_silhouette"] == pytest.approx(b["best_silhouette"])
    assert [s["decision"] for s in a["steps"]] == [s["decision"] for s in b["steps"]]
