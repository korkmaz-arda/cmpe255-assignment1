"""F02/F12/S02/S08 - catalog integrity, total routing, and no dead navigation keys."""

from __future__ import annotations

import pytest

from skills_lab import catalog, execute, status


class _FakeResult:
    def __init__(self, name):
        self.name = name

    def payload(self):
        return {"computed": self.name}


def _fake_results():
    return {
        view.value: _FakeResult(view.value)
        for view in catalog.View
        if view is not catalog.View.CATALOG
    }


def test_every_skill_jump_target_is_a_real_view():
    for skill in catalog.SKILLS:
        assert skill.view in catalog.View
        assert skill.view is not catalog.View.CATALOG
        assert skill.workspace_label  # every target names its dataset


def test_catalog_is_internally_consistent():
    ids = [s.id for s in catalog.SKILLS]
    assert len(ids) == len(set(ids))
    assert catalog.skill_count() == len(catalog.SKILLS)
    for skill in catalog.SKILLS:
        assert skill.category in catalog.CATEGORIES
        assert skill.pitfalls and skill.intuition and skill.purpose


def test_search_matches_name_purpose_and_origin_and_composes_with_category():
    assert any(s.id == "ab-testing" for s in catalog.search("Kohavi"))  # origin
    assert any(s.id == "cohort-analysis" for s in catalog.search("acquisition month"))  # purpose
    assert any(s.id == "pipelines" for s in catalog.search("Pipelines as leakage"))  # name
    filtered = catalog.search("", catalog.CATEGORIES[3])
    assert filtered and all(s.category == catalog.CATEGORIES[3] for s in filtered)
    assert catalog.search("zzzznotaskill") == []


def test_execution_routing_is_total_with_no_canned_success():
    results = _fake_results()
    ab_inputs = {
        "control_visitors": 1000, "control_conversions": 50,
        "treatment_visitors": 1000, "treatment_conversions": 60,
    }
    for skill in catalog.SKILLS:
        payload = execute.execute(skill.id, results, ab_inputs=ab_inputs)
        assert payload["skill"] == skill.name
        assert "result" in payload
        if skill.id == "ab-testing":
            assert payload["result"]["z_score"] is not None
        else:
            assert payload["result"] == {"computed": skill.view.value}


def test_unknown_skill_is_an_error_not_a_success():
    with pytest.raises(execute.UnknownSkillError):
        execute.execute("not-a-skill", _fake_results())


def test_missing_result_is_reported_rather_than_faked():
    with pytest.raises(ValueError):
        execute.execute("exploratory-data-analysis", {})


def test_status_reports_catalog_size_and_dataset_readiness():
    report = status.status()
    assert report["skills_in_catalog"] == catalog.skill_count()
    assert len(report["benchmarks"]) == 5
    assert set(report["datasets"]) == {"titanic", "ames", "creditcard", "retail"}


def test_ab_skill_without_counts_asks_for_them_rather_than_raising_a_guard_error():
    """The A/B skill has no dataset; with no counts it must explain, not leak a guard error."""
    with pytest.raises(execute.NeedsUserCountsError) as raised:
        execute.execute("ab-testing", _fake_results(), ab_inputs=None)
    message = str(raised.value)
    assert "counts you enter" in message
    assert "must be greater than zero" not in message
