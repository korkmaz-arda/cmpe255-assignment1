from datetime import date, timedelta

import pytest

from zenith import ValidationError
from zenith.views import (bucket, filter_tasks, matrix_drop_changes, month_grid, quadrant, sidebar_counts,
                          sort_tasks)

TODAY = date(2026, 9, 16)


def iso(days):
    return (TODAY + timedelta(days=days)).isoformat()


def task(id=1, **kw):
    base = {"id": id, "title": f"t{id}", "description": "", "priority": "medium", "status": "todo",
            "category_id": None, "due_date": None, "order_index": id * 1000, "pinned": False,
            "archived": False, "deleted": False, "created_at": f"2026-09-{id:02d}T00:00:00",
            "completed_at": None, "tags": [], "subtasks": [], "time_spent_minutes": 0}
    base.update(kw)
    return base


@pytest.mark.parametrize("kw,expected", [
    ({"due_date": iso(-3)}, "overdue"), ({"due_date": iso(0)}, "today"), ({"due_date": iso(1)}, "tomorrow"),
    ({"due_date": iso(2)}, "upcoming"), ({}, "no_date"),
    ({"due_date": iso(-3), "status": "completed"}, "completed"), ({"status": "completed"}, "completed"),
])
def test_buckets(kw, expected):
    assert bucket(task(**kw), TODAY) == expected


@pytest.mark.parametrize("kw,expected", [
    ({"priority": "urgent"}, "do_first"),
    ({"priority": "high", "due_date": iso(0)}, "do_first"),
    ({"priority": "high", "due_date": iso(-1)}, "do_first"),
    ({"priority": "high", "due_date": iso(1)}, "schedule"),
    ({"priority": "high"}, "schedule"),
    ({"priority": "medium", "due_date": iso(0)}, "delegate"),
    ({"priority": "low", "due_date": iso(-5)}, "delegate"),
    ({"priority": "medium", "due_date": iso(3)}, "eliminate"),
    ({"priority": "low"}, "eliminate"),
    ({"priority": "urgent", "status": "completed"}, None),
])
def test_quadrant_derivation(kw, expected):
    assert quadrant(task(**kw), TODAY) == expected


DUE_CASES = {"past": iso(-4), "today": iso(0), "future": iso(5), "none": None}
PRIORITIES = ("urgent", "high", "medium", "low")


@pytest.mark.parametrize("target", ["do_first", "schedule", "delegate", "eliminate"])
@pytest.mark.parametrize("due_case", list(DUE_CASES))
@pytest.mark.parametrize("priority", PRIORITIES)
def test_every_drop_lands_in_target_quadrant(target, due_case, priority):
    original = task(priority=priority, due_date=DUE_CASES[due_case], title="keep", description="d",
                    category_id=3, pinned=True, time_spent_minutes=9)
    changes = matrix_drop_changes(original, target, TODAY)
    moved = {**original, **changes}
    assert quadrant(moved, TODAY) == target
    assert set(changes) <= {"priority", "due_date"}  # no unrelated field changes


def test_drop_date_semantics():
    urgent_by_date = task(priority="medium", due_date=iso(-2))
    assert matrix_drop_changes(urgent_by_date, "schedule", TODAY) == {"priority": "high", "due_date": iso(1)}
    assert matrix_drop_changes(urgent_by_date, "eliminate", TODAY) == {"priority": "low", "due_date": None}
    future = task(priority="urgent", due_date=iso(4))
    assert matrix_drop_changes(future, "schedule", TODAY) == {"priority": "high"}
    assert matrix_drop_changes(future, "eliminate", TODAY) == {"priority": "low"}
    assert matrix_drop_changes(future, "do_first", TODAY) == {"priority": "urgent", "due_date": iso(0)}
    assert matrix_drop_changes(future, "delegate", TODAY) == {"priority": "medium", "due_date": iso(0)}
    with pytest.raises(ValidationError):
        matrix_drop_changes(future, "nowhere", TODAY)


def test_store_matrix_drop_uses_domain_rule(store):
    t = store.create_task({"title": "x", "priority": "urgent", "due_date": "2026-09-10", "description": "keep"})
    moved = store.move_to_quadrant(t["id"], "eliminate")
    assert (moved["priority"], moved["due_date"], moved["quadrant"], moved["description"]) == ("low", None, "eliminate", "keep")
    moved = store.move_to_quadrant(t["id"], "do_first")
    assert moved["quadrant"] == "do_first" and moved["due_date"] == "2026-09-16"
    moved = store.move_to_quadrant(t["id"], "schedule")
    assert moved["quadrant"] == "schedule" and moved["due_date"] == "2026-09-17"


def test_priority_sort_uses_severity_not_spelling():
    tasks = [task(1, priority="low"), task(2, priority="high"), task(3, priority="urgent"), task(4, priority="medium")]
    assert [t["priority"] for t in sort_tasks(tasks, "priority")] == ["urgent", "high", "medium", "low"]


@pytest.mark.parametrize("sort", ["custom", "due_asc", "due_desc", "priority", "title", "recent"])
def test_pinned_first_under_every_sort(sort):
    tasks = [task(1, priority="urgent", due_date=iso(0), title="a"), task(2, pinned=True, priority="low", title="z"),
             task(3, due_date=iso(9), title="m"), task(4, pinned=True, title="b", due_date=iso(2))]
    ordered = sort_tasks(tasks, sort)
    assert [t["pinned"] for t in ordered] == [True, True, False, False]


def test_sort_orders():
    tasks = [task(1, due_date=iso(3), title="Beta"), task(2, title="alpha"), task(3, due_date=iso(1), title="Gamma")]
    assert [t["id"] for t in sort_tasks(tasks, "due_asc")] == [3, 1, 2]
    assert [t["id"] for t in sort_tasks(tasks, "due_desc")] == [1, 3, 2]
    assert [t["id"] for t in sort_tasks(tasks, "title")] == [2, 1, 3]
    assert [t["id"] for t in sort_tasks(tasks, "recent")] == [3, 2, 1]
    assert [t["id"] for t in sort_tasks(tasks, "custom")] == [1, 2, 3]


def fixture_set():
    tag_a = {"id": 10, "name": "a", "color": "#000000"}
    return [
        task(1, due_date=iso(0), priority="high", category_id=1, tags=[tag_a], title="Write report"),
        task(2, due_date=iso(5), category_id=1, description="Report appendix"),
        task(3, status="completed", priority="urgent", category_id=2, tags=[tag_a]),
        task(4, archived=True, category_id=1, due_date=iso(0)),
        task(5, deleted=True, category_id=1, tags=[tag_a]),
        task(6, due_date=iso(1)),
    ]


def ids(tasks):
    return [t["id"] for t in tasks]


def test_filters_compose():
    tasks = fixture_set()
    assert ids(filter_tasks(tasks, TODAY)) == [1, 2, 3, 6]
    assert ids(filter_tasks(tasks, TODAY, smart="today")) == [1]
    assert ids(filter_tasks(tasks, TODAY, smart="upcoming")) == [2]  # after tomorrow only
    assert ids(filter_tasks(tasks, TODAY, smart="important")) == [1, 3]
    assert ids(filter_tasks(tasks, TODAY, smart="completed")) == [3]
    assert ids(filter_tasks(tasks, TODAY, category_id=1)) == [1, 2]
    assert ids(filter_tasks(tasks, TODAY, tag_id=10)) == [1, 3]
    assert ids(filter_tasks(tasks, TODAY, search="REPORT")) == [1, 2]
    assert ids(filter_tasks(tasks, TODAY, search="report", category_id=1, tag_id=10, smart="important")) == [1]
    assert ids(filter_tasks(tasks, TODAY, scope="archive")) == [4]
    assert ids(filter_tasks(tasks, TODAY, scope="trash")) == [5]
    assert ids(filter_tasks(tasks, TODAY, scope="trash", tag_id=10)) == [5]
    with pytest.raises(ValidationError):
        filter_tasks(tasks, TODAY, scope="elsewhere")


def test_sidebar_counts_describe_destinations_not_current_view():
    counts = sidebar_counts(fixture_set(), TODAY)
    assert counts["smart"] == {"all": 4, "today": 1, "upcoming": 1, "important": 2, "completed": 1}
    assert counts["archive"] == 1 and counts["trash"] == 1
    assert counts["categories"] == {1: 2, 2: 1}
    assert counts["tags"] == {10: 2}


def test_month_grid_starts_sunday():
    grid = month_grid(2026, 9, TODAY)
    first_week = grid["weeks"][0]
    assert first_week[0]["date"] == "2026-08-30"  # Sunday
    assert first_week[0]["in_month"] is False
    assert all(len(w) == 7 for w in grid["weeks"])
    today_cells = [d for w in grid["weeks"] for d in w if d["is_today"]]
    assert [d["date"] for d in today_cells] == ["2026-09-16"]
    assert grid["prev"] == {"year": 2026, "month": 8} and grid["next"] == {"year": 2026, "month": 10}
    assert month_grid(2026, 12, TODAY)["next"] == {"year": 2027, "month": 1}
