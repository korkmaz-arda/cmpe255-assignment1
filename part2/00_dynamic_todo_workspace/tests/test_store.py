import sqlite3

import pytest

from zenith import NotFoundError, ValidationError
from zenith.db import Database
from zenith.store import Store


def make(store, title="Task", **extra):
    return store.create_task({"title": title, **extra})


# ---------------------------------------------------------------- schema
def test_foreign_keys_enforced_on_every_connection(store):
    with store.db.connect() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO subtasks (task_id, title) VALUES (999, 'orphan')")


def test_enumerations_constrained_at_storage_layer(store):
    task = make(store)
    with store.db.connect() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE tasks SET priority='extreme' WHERE id=?", (task["id"],))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (task["id"],))


def test_validation_rejects_bad_input(store):
    with pytest.raises(ValidationError):
        make(store, "   ")
    with pytest.raises(ValidationError):
        make(store, priority="extreme")
    with pytest.raises(ValidationError):
        make(store, due_date="2026-13-01")
    with pytest.raises(ValidationError):
        make(store, category_id=42)
    with pytest.raises(ValidationError):
        make(store, estimated_minutes=-5)
    with pytest.raises(ValidationError):
        store.create_task({"title": "x", "tag_ids": [99]})


def test_title_trimmed_and_defaults(store):
    task = make(store, "  Hello  ")
    assert task["title"] == "Hello"
    assert (task["priority"], task["status"], task["pinned"], task["time_spent_minutes"]) == ("medium", "todo", False, 0)


def test_persistence_across_reopen(db_path, clock):
    first = Store(Database(db_path), clock)
    cat = first.create_category("Work", "#6366f1")
    tag, _ = first.create_tag("infra")
    task = first.create_task({"title": "Persist me", "category_id": cat["id"], "tag_ids": [tag["id"]],
                              "subtasks": [{"title": "a", "completed": True}, {"title": "b"}]})
    reopened = Store(Database(db_path), clock)
    loaded = reopened.get_task(task["id"])
    assert loaded["title"] == "Persist me"
    assert loaded["category_id"] == cat["id"]
    assert [t["name"] for t in loaded["tags"]] == ["infra"]
    assert [(s["title"], s["completed"]) for s in loaded["subtasks"]] == [("a", True), ("b", False)]


# ------------------------------------------------------------------ seed
def test_seed_runs_once_only(db_path, clock):
    store = Store(Database(db_path), clock)
    assert store.seed_if_first_run() is True
    tasks = store.all_tasks()
    assert len(tasks) == 5
    assert len(store.list_categories()) == 5
    assert {t["name"] for t in store.list_tags()} == {"frontend", "backend", "design", "critical", "learning"}
    assert sum(t["pinned"] for t in tasks) == 2
    completed = [t for t in tasks if t["status"] == "completed"]
    assert len(completed) == 1 and completed[0]["completed_at"] == "2026-09-16T10:00:00"
    assert {t["due_date"] for t in tasks} >= {"2026-09-16", "2026-09-17", "2026-09-23"}

    for t in tasks:
        store.delete_task(t["id"], permanent=True)
    assert Store(Database(db_path), clock).seed_if_first_run() is False
    assert store.all_tasks() == []


# ------------------------------------------------------------ ordering
def test_creation_order_index_is_1000_above_max(store):
    a = make(store, "a")
    b = make(store, "b")
    assert b["order_index"] == a["order_index"] + 1000
    store.reorder([{"id": a["id"], "order_index": 50_000}])
    assert make(store, "c")["order_index"] == 51_000


# ------------------------------------------------ completion lifecycle
def test_completion_timestamp_lifecycle(store, clock):
    task = make(store)
    assert task["completed_at"] is None

    done = store.patch_task(task["id"], {"status": "completed"})
    assert done["completed_at"] == "2026-09-16T10:00:00"

    clock.set("2026-09-16T11:00:00")
    again = store.patch_task(task["id"], {"status": "completed"})  # done -> done keeps stamp
    assert again["completed_at"] == "2026-09-16T10:00:00"

    edited = store.patch_task(task["id"], {"title": "Renamed", "priority": "high"})  # unrelated edit
    assert edited["completed_at"] == "2026-09-16T10:00:00"

    store.reorder([{"id": task["id"], "order_index": 5}], status="completed")
    assert store.get_task(task["id"])["completed_at"] == "2026-09-16T10:00:00"
    store.reorder([{"id": task["id"], "order_index": 7}])
    assert store.get_task(task["id"])["completed_at"] == "2026-09-16T10:00:00"

    reopened = store.patch_task(task["id"], {"status": "in_progress"})
    assert reopened["completed_at"] is None

    clock.set("2026-09-17T09:30:00")
    recompleted = store.patch_task(task["id"], {"status": "completed"})
    assert recompleted["completed_at"] == "2026-09-17T09:30:00"


def test_full_update_preserves_completion_and_replaces_children(store, clock):
    t1, _ = store.create_tag("one")
    t2, _ = store.create_tag("two")
    task = make(store, status="completed", tag_ids=[t1["id"]], subtasks=[{"title": "old"}])
    clock.set("2026-09-18T08:00:00")
    updated = store.update_task(task["id"], {
        "title": "New", "description": "d", "status": "completed", "priority": "low", "category_id": None,
        "due_date": None, "estimated_minutes": 10, "time_spent_minutes": 3, "pinned": True,
        "subtasks": [{"title": "x", "completed": True}, {"title": "y"}], "tag_ids": [t2["id"]]})
    assert updated["completed_at"] == "2026-09-16T10:00:00"
    assert [s["title"] for s in updated["subtasks"]] == ["x", "y"]
    assert [t["name"] for t in updated["tags"]] == ["two"]
    assert updated["pinned"] is True


def test_partial_update_touches_only_supplied_fields(store):
    task = make(store, description="keep", priority="high", due_date="2026-09-20", estimated_minutes=30)
    patched = store.patch_task(task["id"], {"title": "Only title"})
    for key in ("description", "priority", "due_date", "estimated_minutes", "status"):
        assert patched[key] == task[key]
    assert patched["title"] == "Only title"


def test_board_quick_add_into_done_is_stamped(store):
    task = store.create_from_capture("Already done", status="completed")
    assert task["completed_at"] == "2026-09-16T10:00:00"


def test_reorder_with_status_moves_batch(store):
    a, b = make(store, "a"), make(store, "b", status="completed")
    moved = store.reorder([{"id": a["id"], "order_index": 1}, {"id": b["id"], "order_index": 2}], status="review")
    assert [(t["status"], t["completed_at"]) for t in moved] == [("review", None), ("review", None)]


# ------------------------------------------------------------- capture
def test_capture_resolves_tag_names_creating_or_reusing(store):
    existing, _ = store.create_tag("backend", "#123456")
    task = store.create_from_capture("Fix API !u #Backend #newtag ~1.5h @fri")
    names = {t["name"]: t for t in task["tags"]}
    assert set(names) == {"backend", "newtag"}
    assert names["backend"]["id"] == existing["id"]
    assert (task["title"], task["priority"], task["estimated_minutes"], task["due_date"]) == (
        "Fix API", "urgent", 90, "2026-09-18")
    assert len(store.list_tags()) == 2


def test_capture_calendar_default_due_and_override(store):
    plain = store.create_from_capture("On clicked day", default_due="2026-10-05")
    assert plain["due_date"] == "2026-10-05"
    override = store.create_from_capture("Explicit @tomorrow", default_due="2026-10-05")
    assert override["due_date"] == "2026-09-17"
    unknown = store.create_from_capture("Unknown token @someday", default_due="2026-10-05")
    assert unknown["due_date"] == "2026-10-05"
    assert unknown["title"] == "Unknown token"


def test_capture_requires_residual_title(store):
    with pytest.raises(ValidationError):
        store.create_from_capture("!high #tag @today")


# ------------------------------------------------------------- deletion
def test_two_stage_delete_and_restore(store):
    task = store.create_task({"title": "t", "subtasks": [{"title": "s"}], "tag_names": ["x"]})
    assert store.delete_task(task["id"]) == "trashed"
    assert store.get_task(task["id"])["deleted"] is True
    store.restore_task(task["id"])
    assert store.get_task(task["id"])["deleted"] is False
    store.delete_task(task["id"])
    assert store.delete_task(task["id"]) == "purged"
    with pytest.raises(NotFoundError):
        store.get_task(task["id"])
    with store.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM subtasks").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_tags").fetchone()[0] == 0
    assert [t["name"] for t in store.list_tags()] == ["x"]  # shared vocabulary survives


def test_explicit_permanent_delete(store):
    task = make(store)
    assert store.delete_task(task["id"], permanent=True) == "purged"


def test_delete_category_detaches_tasks(store):
    cat = store.create_category("Temp")
    task = make(store, category_id=cat["id"])
    store.delete_category(cat["id"])
    kept = store.get_task(task["id"])
    assert kept["category_id"] is None and kept["deleted"] is False


def test_delete_tag_removes_links_not_tasks(store):
    tag, _ = store.create_tag("gone")
    task = make(store, tag_ids=[tag["id"]])
    store.delete_tag(tag["id"])
    kept = store.get_task(task["id"])
    assert kept["tags"] == [] and kept["deleted"] is False


def test_archive_unarchive(store):
    task = make(store)
    assert store.archive_task(task["id"])["archived"] is True
    assert store.unarchive_task(task["id"])["archived"] is False


# ----------------------------------------------------------------- tags
def test_tag_normalization_and_duplicate_reuse(store):
    tag, created = store.create_tag("  #FrontEnd ")
    assert tag["name"] == "frontend" and created
    again, created_again = store.create_tag("frontend")
    assert again["id"] == tag["id"] and not created_again
    with pytest.raises(ValidationError):
        store.create_tag("#")


def test_tag_usage_counts(store):
    tag, _ = store.create_tag("x")
    make(store, tag_ids=[tag["id"]])
    trashed = make(store, tag_ids=[tag["id"]])
    store.delete_task(trashed["id"])
    assert store.list_tags()[0]["usage"] == 1


# ----------------------------------------------------------------- bulk
def test_bulk_actions(store):
    cat = store.create_category("C")
    ids = [make(store, f"t{i}")["id"] for i in range(3)]
    assert store.bulk("complete", ids) == 3
    assert all(store.get_task(i)["completed_at"] for i in ids)
    store.bulk("uncomplete", ids[:1])
    assert store.get_task(ids[0])["completed_at"] is None
    store.bulk("priority", ids, "urgent")
    store.bulk("category", ids, cat["id"])
    assert all(store.get_task(i)["priority"] == "urgent" and store.get_task(i)["category_id"] == cat["id"] for i in ids)
    store.bulk("archive", ids[:1])
    store.bulk("unarchive", ids[:1])
    store.bulk("trash", ids)
    store.bulk("restore", ids[1:])
    store.bulk("purge", ids[:1])
    assert [t["id"] for t in store.all_tasks()] == ids[1:]
    with pytest.raises(ValidationError):
        store.bulk("priority", ids[1:], "extreme")
    with pytest.raises(ValidationError):
        store.bulk("explode", ids[1:])


# ------------------------------------------------------------ focus time
def test_focus_minutes_accumulate(store):
    task = make(store, time_spent_minutes=5)
    assert store.add_focus_minutes(task["id"], 25)["time_spent_minutes"] == 30
    assert store.add_focus_minutes(task["id"], 7)["time_spent_minutes"] == 37
    for bad in (0, -1, 2000, 1.5, True):
        with pytest.raises(ValidationError):
            store.add_focus_minutes(task["id"], bad)


# ------------------------------------------------------------- activity
def test_every_mutation_is_audited(store, clock):
    task = make(store, "Audited")
    store.patch_task(task["id"], {"status": "completed"})
    store.patch_task(task["id"], {"priority": "low"})
    store.archive_task(task["id"])
    store.delete_task(task["id"])
    store.delete_task(task["id"])
    store.bulk("complete", [make(store, "b")["id"]])
    log = store.activity(limit=50)
    actions = [e["action"] for e in log]
    assert actions[:1] == ["batch operation"]
    for expected in ("created", "completed", "updated", "archived", "moved to trash", "permanently deleted"):
        assert expected in actions
    assert all(e["created_at"] for e in log)
    assert log[-1]["task_title"] == "Audited" and log[-1]["action"] == "created"
    assert len(store.activity(limit=2)) == 2
    assert store.clear_activity() == len(log)
    assert store.activity() == []


def test_activity_details_list_only_actual_changes(store):
    tag, _ = store.create_tag("t")
    task = make(store, "Same", priority="high", subtasks=[{"title": "s"}])
    full = {"title": "Same", "description": "", "status": "todo", "priority": "high", "category_id": None,
            "due_date": None, "estimated_minutes": None, "time_spent_minutes": 0, "pinned": False,
            "subtasks": [{"title": "s", "completed": False}], "tag_ids": []}
    store.update_task(task["id"], full)
    assert store.activity(1)[0]["details"] == "saved with no changes"
    store.update_task(task["id"], {**full, "priority": "low", "tag_ids": [tag["id"]]})
    assert store.activity(1)[0]["details"] == "priority, tags"
    store.update_task(task["id"], {**full, "priority": "low", "tag_ids": [tag["id"]],
                                   "subtasks": [{"title": "s", "completed": True}]})
    assert store.activity(1)[0]["details"] == "subtasks"
    store.patch_task(task["id"], {"status": "completed"})
    assert store.activity(1)[0]["action"] == "completed"
    store.patch_task(task["id"], {"status": "completed"})
    assert store.activity(1)[0]["details"] == "saved with no changes"
    store.patch_task(task["id"], {"status": "todo", "title": "Renamed"})
    assert store.activity(1)[0]["details"] == "reopened, title"
