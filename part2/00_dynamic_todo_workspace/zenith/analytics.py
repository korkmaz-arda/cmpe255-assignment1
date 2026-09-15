"""Descriptive productivity telemetry.

Recomputed from the current task store on every call; nothing is cached or
persisted. Archived and trashed tasks are excluded from every aggregate
(including the heatmap). Completion history is read from the tasks' current
``completed_at`` stamps, so reopening a task removes that day's completion.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from . import PRIORITIES, STATUSES
from .views import in_scope, is_overdue

WINDOW_DAYS = 30
UNCATEGORIZED = {"id": None, "name": "Uncategorized", "color": "#64748b"}


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def _pct(part: int, whole: int) -> int:
    return _round_half_up(part / whole * 100) if whole else 0


def streak_from(completions_by_day: list[int]) -> int:
    """Consecutive days with >=1 completion, counted back from the last entry (today).

    A zero *today* does not break the streak (counting starts yesterday);
    any earlier zero day ends it. Bounded by the window length.
    """
    days = list(reversed(completions_by_day))
    if days and days[0] == 0:
        days = days[1:]
    streak = 0
    for count in days:
        if count == 0:
            break
        streak += 1
    return streak


def productivity_index(completion_rate: int, streak: int, overdue: int) -> int:
    """Ad-hoc heuristic composite (not a validated metric), clamped to 10-100."""
    raw = _round_half_up(completion_rate * 0.7) + min(streak * 5, 20) - min(overdue * 10, 30) + 20
    return max(10, min(100, raw))


def heat_level(completed: int, created: int) -> int:
    """0 = no activity; 1 = tasks created but none completed; 2/3/4 = >=1/>=2/>=4 completions."""
    if completed >= 4:
        return 4
    if completed >= 2:
        return 3
    if completed >= 1:
        return 2
    if created >= 1:
        return 1
    return 0


def compute(tasks: list[dict], categories: list[dict], today: date) -> dict:
    active = [t for t in tasks if in_scope(t, "active")]
    total = len(active)
    completed = sum(1 for t in active if t["status"] == "completed")
    rate = _pct(completed, total)
    overdue = sum(1 for t in active if is_overdue(t, today))

    window = [today - timedelta(days=offset) for offset in range(WINDOW_DAYS - 1, -1, -1)]
    done_per_day = {d.isoformat(): 0 for d in window}
    created_per_day = {d.isoformat(): 0 for d in window}
    for t in active:
        if t["status"] == "completed" and t["completed_at"]:
            day = t["completed_at"][:10]
            if day in done_per_day:
                done_per_day[day] += 1
        day = t["created_at"][:10]
        if day in created_per_day:
            created_per_day[day] += 1
    heatmap = [{"date": d, "completed": done_per_day[d], "created": created_per_day[d],
                "level": heat_level(done_per_day[d], created_per_day[d])} for d in done_per_day]
    streak = streak_from([cell["completed"] for cell in heatmap])

    focus_minutes = sum(t["time_spent_minutes"] for t in active)

    priorities = []
    for p in PRIORITIES:
        subset = [t for t in active if t["priority"] == p]
        done = sum(1 for t in subset if t["status"] == "completed")
        priorities.append({"priority": p, "total": len(subset), "completed": done,
                           "completion_pct": _pct(done, len(subset)), "share_pct": _pct(len(subset), total)})

    by_category = []
    known_ids = set()
    for c in categories:
        known_ids.add(c["id"])
        subset = [t for t in active if t["category_id"] == c["id"]]
        by_category.append({"id": c["id"], "name": c["name"], "color": c["color"], "total": len(subset),
                            "completed": sum(1 for t in subset if t["status"] == "completed")})
    uncategorized = [t for t in active if t["category_id"] not in known_ids]
    if uncategorized:
        by_category.append({**UNCATEGORIZED, "total": len(uncategorized),
                            "completed": sum(1 for t in uncategorized if t["status"] == "completed")})

    return {
        "today": today.isoformat(),
        "total": total,
        "completed": completed,
        "completion_rate": rate,
        "streak": streak,
        "overdue": overdue,
        "productivity_index": productivity_index(rate, streak, overdue),
        "focus_minutes": focus_minutes,
        "focus_hours": _round_half_up(focus_minutes / 6) / 10,
        "status_counts": {s: sum(1 for t in active if t["status"] == s) for s in STATUSES},
        "heatmap": heatmap,
        "priorities": priorities,
        "categories": by_category,
    }
