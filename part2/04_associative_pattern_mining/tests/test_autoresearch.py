"""The parameter search: real trials, a real acceptance rule, honest rejections."""

from __future__ import annotations

import json

import pytest

from basket import autoresearch, config


@pytest.fixture
def searched(prepared, monkeypatch):
    # The fixture is tiny, so the search needs thresholds it can actually clear.
    monkeypatch.setattr(config, "SEARCH_BASELINE",
                        {"min_support": 0.05, "max_len": 4,
                         "min_confidence": 0.2, "min_lift": 1.0})
    monkeypatch.setattr(config, "BUNDLE_MIN_SUPPORT", 0.10)
    monkeypatch.setattr(config, "PHASE2_MIN_RULES", 2)
    monkeypatch.setattr(config, "PHASE3_MIN_RULES", 2)
    return autoresearch.execute(n_orders=105)


def test_all_four_phases_run(searched):
    phases = {step["phase"].split("—")[0].strip() for step in searched["trajectory"]}
    assert phases == {"Phase 1", "Phase 2", "Phase 3", "Phase 4"}
    assert searched["total_iterations"] == 11        # 1 baseline + 5 + 4 + 1


def test_backbone_tournament_ranks_by_measured_runtime(searched):
    times = [row["elapsed_seconds"] for row in searched["tournament"]]
    assert times == sorted(times)
    assert searched["tournament"][0]["is_champion"]
    # All three are exact, so the tournament picks a runtime, not a result.
    assert len({row["n_itemsets"] for row in searched["tournament"]}) == 1


def test_every_step_records_the_full_trial(searched):
    required = {
        "step_id", "iteration", "phase", "category", "hypothesis", "code_change",
        "params", "mean_lift_before", "mean_lift_after", "delta", "decision",
        "reflection", "timestamp", "n_rules",
    }
    for step in searched["trajectory"]:
        assert required <= set(step)
        assert step["delta"] == pytest.approx(step["mean_lift_after"] - step["mean_lift_before"])


def test_acceptance_requires_a_real_gain(searched):
    for step in searched["trajectory"]:
        if step["decision"] == "accepted":
            assert step["delta"] > config.MIN_LIFT_GAIN
    assert searched["best_mean_lift"] >= searched["initial_mean_lift"]


def test_rule_floor_rejects_a_thin_rule_set():
    """Mean lift is trivially gamed by keeping three freak rules; the floor stops it."""
    outcome = {"mean_lift": 99.0, "n_rules": 3, "n_itemsets": 5, "top_lift": 99.0,
               "mean_confidence": 1.0, "elapsed_seconds": 0.01, "max_antecedent_len": 1}
    decision, reflection = autoresearch._verdict(1.0, outcome, min_rules=10)
    assert decision == "rejected"
    assert "floor" in reflection

    outcome["n_rules"] = 50
    decision, _ = autoresearch._verdict(1.0, outcome, min_rules=10)
    assert decision == "accepted"


def test_a_gain_below_the_threshold_is_rejected():
    outcome = {"mean_lift": 1.01, "n_rules": 500, "n_itemsets": 50, "top_lift": 3.0,
               "mean_confidence": 0.5, "elapsed_seconds": 0.01, "max_antecedent_len": 2}
    decision, reflection = autoresearch._verdict(1.0, outcome, min_rules=10)
    assert decision == "rejected"
    assert "threshold" in reflection


def test_rejected_trials_are_kept(searched):
    decisions = [step["decision"] for step in searched["trajectory"]]
    assert "rejected" in decisions
    # Step 0 is the baseline measurement, not a trial with a verdict.
    assert decisions.count("baseline") == 1
    assert searched["accepted"] + searched["rejected"] == len(searched["trajectory"]) - 1


def test_phase_four_lifts_the_length_cap(searched):
    """A 4-product antecedent needs a 5-item itemset, so this phase must mine deeper."""
    phase4 = [s for s in searched["trajectory"] if s["phase"].startswith("Phase 4")]
    assert len(phase4) == 1
    assert phase4[0]["params"]["max_len"] >= 5
    assert phase4[0]["max_antecedent_len"] == 4
    assert all(s["params"]["max_len"] <= 4
               for s in searched["trajectory"] if not s["phase"].startswith("Phase 4")
               and s["category"] != "itemset length")


def test_summary_matches_the_trajectory(searched):
    assert searched["accepted"] == sum(
        1 for s in searched["trajectory"] if s["decision"] == "accepted"
    )
    if searched["initial_mean_lift"]:
        expected = ((searched["best_mean_lift"] - searched["initial_mean_lift"])
                    / searched["initial_mean_lift"] * 100.0)
        assert searched["improvement_pct"] == pytest.approx(expected)


def test_search_record_persists(searched):
    autoresearch.save(searched)
    assert config.AUTORESEARCH_JSON.exists()
    payload = json.loads(config.AUTORESEARCH_JSON.read_text())
    assert payload["objective"] == "mean lift across the surviving rule set"
    assert "rule" in payload["acceptance_rule"]
