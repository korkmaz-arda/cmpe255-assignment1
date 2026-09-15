from datetime import date

import pytest

from zenith import analytics
from zenith.analytics import heat_level, productivity_index, streak_from

TODAY = date(2026, 9, 16)


def task(id, **kw):
    base = {"id": id, "title": "t", "description": "", "priority": "medium", "status": "todo", "category_id": None,
            "due_date": None, "order_index": 0, "pinned": False, "archived": False, "deleted": False,
            "created_at": "2026-09-16T09:00:00", "completed_at": None, "tags": [], "subtasks": [],
            "time_spent_minutes": 0}
    base.update(kw)
    return base


def done(id, day, **kw):
    return task(id, status="completed", completed_at=f"{day}T12:00:00", **kw)


@pytest.mark.parametrize("days,expected", [
    ([0] * 30, 0),
    ([0] * 29 + [1], 1),
    ([0] * 27 + [1, 1, 1], 3),
    ([0] * 27 + [1, 1, 0], 2),  # zero today does not break the streak
    ([0] * 27 + [1, 0, 1], 1),  # an earlier zero day ends it
    ([0] * 27 + [1, 0, 0], 0),
    ([2] * 30, 30),
    ([2] * 29 + [0], 29),
])
def test_streak(days, expected):
    assert streak_from(days) == expected


@pytest.mark.parametrize("rate,streak,overdue,expected", [
    (0, 0, 0, 20), (100, 0, 0, 90), (100, 4, 0, 100), (100, 10, 0, 100),
    (0, 0, 5, 10), (50, 2, 1, 55), (15, 0, 0, 31),  # round(10.5) -> 11 (half up)
])
def test_productivity_index(rate, streak, overdue, expected):
    assert productivity_index(rate, streak, overdue) == expected


@pytest.mark.parametrize("completed,created,level", [(0, 0, 0), (0, 3, 1), (1, 0, 2), (2, 0, 3), (3, 9, 3), (4, 0, 4)])
def test_heat_levels(completed, created, level):
    assert heat_level(completed, created) == level


def test_compute_end_to_end():
    categories = [{"id": 1, "name": "Work", "color": "#111111"}, {"id": 2, "name": "Empty", "color": "#222222"}]
    tasks = [
        done(1, "2026-09-16", priority="high", category_id=1, time_spent_minutes=50),
        done(2, "2026-09-15", priority="high", category_id=1, created_at="2026-09-15T08:00:00"),
        done(3, "2026-09-14", priority="low"),
        task(4, priority="urgent", due_date="2026-09-10", time_spent_minutes=25),  # overdue
        task(5, priority="high", due_date="2026-09-16"),  # due today, not overdue
        done(6, "2026-09-16", deleted=True, time_spent_minutes=999),  # excluded everywhere
        done(7, "2026-09-16", archived=True),  # excluded everywhere
        done(8, "2026-08-01", created_at="2026-07-01T00:00:00"),  # outside window
    ]
    a = analytics.compute(tasks, categories, TODAY)
    assert (a["total"], a["completed"], a["completion_rate"]) == (6, 4, 67)
    assert a["overdue"] == 1
    assert a["streak"] == 3
    assert a["productivity_index"] == productivity_index(67, 3, 1) == 47 + 15 - 10 + 20
    assert (a["focus_minutes"], a["focus_hours"]) == (75, 1.3)
    assert a["status_counts"] == {"todo": 2, "in_progress": 0, "review": 0, "completed": 4}

    heat = a["heatmap"]
    assert len(heat) == 30
    assert heat[0]["date"] == "2026-08-18" and heat[-1]["date"] == "2026-09-16"
    by_day = {c["date"]: c for c in heat}
    assert by_day["2026-09-16"]["completed"] == 1  # deleted/archived completions not counted
    assert by_day["2026-09-16"]["created"] == 4
    assert by_day["2026-09-15"]["completed"] == 1 and by_day["2026-09-15"]["created"] == 1
    assert by_day["2026-09-01"] == {"date": "2026-09-01", "completed": 0, "created": 0, "level": 0}

    prio = {p["priority"]: p for p in a["priorities"]}
    assert prio["high"] == {"priority": "high", "total": 3, "completed": 2, "completion_pct": 67, "share_pct": 50}
    assert prio["medium"]["total"] == 1 and prio["medium"]["completion_pct"] == 100
    assert prio["urgent"]["completion_pct"] == 0

    cats = {c["name"]: c for c in a["categories"]}
    assert (cats["Work"]["completed"], cats["Work"]["total"]) == (2, 2)
    assert (cats["Empty"]["completed"], cats["Empty"]["total"]) == (0, 0)
    assert (cats["Uncategorized"]["completed"], cats["Uncategorized"]["total"]) == (2, 4)


def test_empty_store():
    a = analytics.compute([], [], TODAY)
    assert (a["total"], a["completion_rate"], a["streak"], a["productivity_index"]) == (0, 0, 0, 20)
    assert len(a["heatmap"]) == 30 and all(c["level"] == 0 for c in a["heatmap"])
    assert a["categories"] == []


def test_reopening_removes_completion_from_telemetry(store):
    t = store.create_task({"title": "x"})
    store.patch_task(t["id"], {"status": "completed"})
    assert analytics.compute(store.all_tasks(), [], store.today())["heatmap"][-1]["completed"] == 1
    store.patch_task(t["id"], {"status": "todo"})
    a = analytics.compute(store.all_tasks(), [], store.today())
    assert a["heatmap"][-1]["completed"] == 0 and a["completed"] == 0
