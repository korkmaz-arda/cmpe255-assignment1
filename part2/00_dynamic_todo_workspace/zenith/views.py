"""Derived classifications, filtering, sorting, and navigation counts.

All of these are computed from stored fields at read time and never
persisted: list date buckets, Eisenhower urgency/importance/quadrant,
overdue status, smart-list membership, and sidebar counts.

Tasks are plain dicts as produced by ``Store`` (``due_date`` is an ISO
string or ``None``).
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta

from . import PRIORITY_RANK, ValidationError

BUCKETS = ("overdue", "today", "tomorrow", "upcoming", "no_date", "completed")
QUADRANTS = ("do_first", "schedule", "delegate", "eliminate")
SCOPES = ("active", "archive", "trash")
SMART_LISTS = ("all", "today", "upcoming", "important", "completed")
SORTS = ("custom", "due_asc", "due_desc", "priority", "title", "recent")


def _due(task: dict) -> date | None:
    value = task.get("due_date")
    return date.fromisoformat(value) if value else None


def is_completed(task: dict) -> bool:
    return task["status"] == "completed"


def is_overdue(task: dict, today: date) -> bool:
    due = _due(task)
    return not is_completed(task) and due is not None and due < today


def bucket(task: dict, today: date) -> str:
    if is_completed(task):
        return "completed"
    due = _due(task)
    if due is None:
        return "no_date"
    if due < today:
        return "overdue"
    if due == today:
        return "today"
    if due == today + timedelta(days=1):
        return "tomorrow"
    return "upcoming"


# --- Eisenhower matrix -----------------------------------------------------

def is_urgent(task: dict, today: date) -> bool:
    due = _due(task)
    return task["priority"] == "urgent" or (due is not None and due <= today)


def is_important(task: dict) -> bool:
    return task["priority"] in ("urgent", "high")


def quadrant(task: dict, today: date) -> str | None:
    if is_completed(task):
        return None
    urgent, important = is_urgent(task, today), is_important(task)
    if urgent and important:
        return "do_first"
    if important:
        return "schedule"
    if urgent:
        return "delegate"
    return "eliminate"


def matrix_drop_changes(task: dict, target: str, today: date) -> dict:
    """Minimal attribute changes that place ``task`` into ``target``.

    Do First -> priority urgent, due today.
    Schedule -> priority high; a due date of today or earlier moves to tomorrow.
    Delegate -> priority medium, due today.
    Eliminate -> priority low; a due date of today or earlier is cleared.
    Non-urgent due dates are left untouched; no other field changes.
    """
    if target not in QUADRANTS:
        raise ValidationError(f"Unknown quadrant: {target!r}")
    due = _due(task)
    date_is_urgent = due is not None and due <= today
    if target == "do_first":
        return {"priority": "urgent", "due_date": today.isoformat()}
    if target == "delegate":
        return {"priority": "medium", "due_date": today.isoformat()}
    if target == "schedule":
        changes = {"priority": "high"}
        if date_is_urgent:
            changes["due_date"] = (today + timedelta(days=1)).isoformat()
        return changes
    changes = {"priority": "low"}
    if date_is_urgent:
        changes["due_date"] = None
    return changes


# --- Filtering and sorting --------------------------------------------------

def in_scope(task: dict, scope: str) -> bool:
    if scope == "trash":
        return bool(task["deleted"])
    if scope == "archive":
        return bool(task["archived"]) and not task["deleted"]
    return not task["archived"] and not task["deleted"]


def in_smart_list(task: dict, smart: str, today: date) -> bool:
    due = _due(task)
    if smart == "today":
        return due == today
    if smart == "upcoming":
        return due is not None and due > today + timedelta(days=1)
    if smart == "important":
        return is_important(task)
    if smart == "completed":
        return is_completed(task)
    return True


def filter_tasks(tasks: list[dict], today: date, *, scope: str = "active", smart: str = "all",
                 category_id: int | None = None, tag_id: int | None = None,
                 search: str = "") -> list[dict]:
    if scope not in SCOPES:
        raise ValidationError(f"Unknown scope: {scope!r}")
    if smart not in SMART_LISTS:
        raise ValidationError(f"Unknown smart list: {smart!r}")
    term = (search or "").strip().casefold()
    result = []
    for task in tasks:
        if not in_scope(task, scope):
            continue
        if not in_smart_list(task, smart, today):
            continue
        if category_id is not None and task["category_id"] != category_id:
            continue
        if tag_id is not None and tag_id not in {t["id"] for t in task["tags"]}:
            continue
        if term and term not in task["title"].casefold() and term not in (task["description"] or "").casefold():
            continue
        result.append(task)
    return result


def sort_tasks(tasks: list[dict], sort: str = "custom") -> list[dict]:
    """Sort by the chosen key; pinned tasks always come first."""
    if sort not in SORTS:
        raise ValidationError(f"Unknown sort: {sort!r}")
    far = "9999-12-31"
    keyed = list(tasks)
    if sort == "custom":
        keyed.sort(key=lambda t: (t["order_index"], t["id"]))
    elif sort == "due_asc":
        keyed.sort(key=lambda t: (t["due_date"] is None, t["due_date"] or far, t["order_index"], t["id"]))
    elif sort == "due_desc":
        keyed.sort(key=lambda t: (t["order_index"], t["id"]))
        keyed.sort(key=lambda t: t["due_date"] or "", reverse=True)
        keyed.sort(key=lambda t: t["due_date"] is None)
    elif sort == "priority":
        keyed.sort(key=lambda t: (PRIORITY_RANK[t["priority"]], t["order_index"], t["id"]))
    elif sort == "title":
        keyed.sort(key=lambda t: (t["title"].casefold(), t["id"]))
    elif sort == "recent":
        keyed.sort(key=lambda t: (t["created_at"], t["id"]), reverse=True)
    keyed.sort(key=lambda t: not t["pinned"])  # stable: preserves chosen order within groups
    return keyed


def decorate(task: dict, today: date) -> dict:
    """Attach server-derived attributes the client renders but never computes."""
    subtasks = task.get("subtasks", [])
    due = _due(task)
    return {
        **task,
        "bucket": bucket(task, today),
        "quadrant": quadrant(task, today),
        "overdue": is_overdue(task, today),
        "due_today": due == today,
        "subtasks_done": sum(1 for s in subtasks if s["completed"]),
        "subtasks_total": len(subtasks),
    }


def sidebar_counts(tasks: list[dict], today: date) -> dict:
    """Counts for every navigation destination, independent of the current view."""
    active = [t for t in tasks if in_scope(t, "active")]
    categories: dict[int, int] = {}
    tags: dict[int, int] = {}
    for t in active:
        if t["category_id"] is not None:
            categories[t["category_id"]] = categories.get(t["category_id"], 0) + 1
        for tag in t["tags"]:
            tags[tag["id"]] = tags.get(tag["id"], 0) + 1
    return {
        "smart": {s: sum(1 for t in active if in_smart_list(t, s, today)) for s in SMART_LISTS},
        "archive": sum(1 for t in tasks if in_scope(t, "archive")),
        "trash": sum(1 for t in tasks if in_scope(t, "trash")),
        "categories": categories,
        "tags": tags,
    }


def month_grid(year: int, month: int, today: date) -> dict:
    """Month grid with weeks starting on Sunday, including leading/trailing days."""
    if not 1 <= month <= 12 or not 1 <= year <= 9998:
        raise ValidationError("Invalid month")
    cal = calendar.Calendar(firstweekday=6)
    weeks = [
        [{"date": d.isoformat(), "day": d.day, "in_month": d.month == month, "is_today": d == today}
         for d in week]
        for week in cal.monthdatescalendar(year, month)
    ]
    prev_y, prev_m = (year - 1, 12) if month == 1 else (year, month - 1)
    next_y, next_m = (year + 1, 1) if month == 12 else (year, month + 1)
    return {
        "year": year, "month": month, "label": date(year, month, 1).strftime("%B %Y"),
        "weeks": weeks, "prev": {"year": prev_y, "month": prev_m},
        "next": {"year": next_y, "month": next_m},
        "today": today.isoformat(),
    }
