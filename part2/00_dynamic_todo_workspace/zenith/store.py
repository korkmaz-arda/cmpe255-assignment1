"""Task store: every state-changing operation and its validation.

The store is the single source of truth. It enforces the completion
timestamp lifecycle, the soft-delete/purge lifecycle, tag-name
normalization, and records an audit entry for every task mutation.
"""

from __future__ import annotations

import hashlib
import random
import sqlite3
from datetime import date, timedelta

from . import PRIORITIES, STATUSES, NotFoundError, ValidationError
from .capture import parse_capture
from .clock import SystemClock, stamp
from .db import Database
from .views import decorate, matrix_drop_changes

CATEGORY_COLORS = ("#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e",
                   "#8b5cf6", "#ec4899", "#14b8a6", "#84cc16", "#f97316")
TAG_COLORS = ("#6366f1", "#8b5cf6", "#ec4899", "#f43f5e", "#f97316",
              "#f59e0b", "#84cc16", "#10b981", "#06b6d4", "#64748b")
CATEGORY_ICONS = ("folder", "briefcase", "home", "code", "heart", "wallet", "book", "star")

MAX_TITLE = 500
MAX_MINUTES = 100_000
BULK_ACTIONS = ("complete", "uncomplete", "priority", "category", "trash",
                "restore", "purge", "archive", "unarchive")

EDITABLE_FIELDS = ("title", "description", "priority", "status", "category_id", "due_date",
                   "estimated_minutes", "time_spent_minutes", "pinned")


def normalize_tag_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValidationError("Tag name must be text")
    cleaned = name.strip().lstrip("#").strip().lower()
    if not cleaned:
        raise ValidationError("Tag name is required")
    if len(cleaned) > 50:
        raise ValidationError("Tag name is too long")
    return cleaned


def _default_tag_color(name: str) -> str:
    digest = hashlib.sha1(name.encode()).digest()
    return TAG_COLORS[digest[0] % len(TAG_COLORS)]


def _is_color(value) -> bool:
    return isinstance(value, str) and len(value) == 7 and value.startswith("#") and all(
        c in "0123456789abcdefABCDEF" for c in value[1:])


class Store:
    def __init__(self, db: Database, clock=None, rng: random.Random | None = None):
        self.db = db
        self.clock = clock or SystemClock()
        self.rng = rng or random.Random()

    # ------------------------------------------------------------------ utils
    def _now(self) -> str:
        return stamp(self.clock.now())

    def today(self) -> date:
        return self.clock.today()

    @staticmethod
    def _log(conn, task_id: int | None, title: str, action: str, details: str, when: str) -> None:
        conn.execute(
            "INSERT INTO activity_log (task_id, task_title, action, details, created_at) VALUES (?,?,?,?,?)",
            (task_id, title, action, details, when))

    # ------------------------------------------------------------- validation
    def _clean_fields(self, conn, data: dict, *, partial: bool) -> dict:
        unknown = set(data) - set(EDITABLE_FIELDS)
        if unknown:
            raise ValidationError(f"Unknown field(s): {', '.join(sorted(unknown))}")
        out: dict = {}
        if "title" in data or not partial:
            title = data.get("title")
            if not isinstance(title, str) or not title.strip():
                raise ValidationError("Title is required")
            if len(title.strip()) > MAX_TITLE:
                raise ValidationError("Title is too long")
            out["title"] = title.strip()
        if "description" in data:
            desc = data["description"]
            if desc is None:
                desc = ""
            if not isinstance(desc, str):
                raise ValidationError("Description must be text")
            out["description"] = desc
        if "priority" in data:
            if data["priority"] not in PRIORITIES:
                raise ValidationError("Invalid priority")
            out["priority"] = data["priority"]
        if "status" in data:
            if data["status"] not in STATUSES:
                raise ValidationError("Invalid status")
            out["status"] = data["status"]
        if "category_id" in data:
            cid = data["category_id"]
            if cid in ("", None):
                cid = None
            else:
                if isinstance(cid, bool) or not isinstance(cid, int):
                    raise ValidationError("Invalid category")
                if not conn.execute("SELECT 1 FROM categories WHERE id=?", (cid,)).fetchone():
                    raise ValidationError("Category does not exist")
            out["category_id"] = cid
        if "due_date" in data:
            due = data["due_date"]
            if due in ("", None):
                due = None
            else:
                try:
                    due = date.fromisoformat(due).isoformat() if isinstance(due, str) and len(due) == 10 else None
                except ValueError:
                    due = None
                if due is None:
                    raise ValidationError("Due date must be YYYY-MM-DD")
            out["due_date"] = due
        for key in ("estimated_minutes", "time_spent_minutes"):
            if key in data:
                value = data[key]
                if value in ("", None):
                    value = None if key == "estimated_minutes" else 0
                elif isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_MINUTES:
                    raise ValidationError(f"{key.replace('_', ' ').capitalize()} must be a whole number of minutes")
                out[key] = value
        if "pinned" in data:
            if not isinstance(data["pinned"], bool):
                raise ValidationError("Pinned must be true or false")
            out["pinned"] = int(data["pinned"])
        return out

    @staticmethod
    def _clean_subtasks(subtasks) -> list[dict]:
        if not isinstance(subtasks, list):
            raise ValidationError("Subtasks must be a list")
        cleaned = []
        for item in subtasks:
            if not isinstance(item, dict) or not isinstance(item.get("title"), str) or not item["title"].strip():
                raise ValidationError("Each subtask needs a title")
            cleaned.append({"title": item["title"].strip()[:MAX_TITLE], "completed": bool(item.get("completed"))})
        return cleaned

    @staticmethod
    def _clean_tag_ids(conn, tag_ids) -> list[int]:
        if not isinstance(tag_ids, list) or any(isinstance(t, bool) or not isinstance(t, int) for t in tag_ids):
            raise ValidationError("Tags must be a list of tag ids")
        unique = list(dict.fromkeys(tag_ids))
        for tid in unique:
            if not conn.execute("SELECT 1 FROM tags WHERE id=?", (tid,)).fetchone():
                raise ValidationError("Tag does not exist")
        return unique

    def _completion_after(self, old_status: str | None, old_completed_at: str | None, new_status: str) -> str | None:
        """Completion timestamp lifecycle.

        not done -> done: stamp now; done -> done: keep the existing stamp;
        anything -> not done: clear.
        """
        if new_status != "completed":
            return None
        if old_status == "completed" and old_completed_at:
            return old_completed_at
        return self._now()

    # ------------------------------------------------------------------ reads
    def _load_tasks(self, conn, where: str = "", params: tuple = ()) -> list[dict]:
        rows = conn.execute(f"SELECT * FROM tasks {where}", params).fetchall()
        tasks = {r["id"]: {**dict(r), "pinned": bool(r["pinned"]), "archived": bool(r["archived"]),
                           "deleted": bool(r["deleted"]), "subtasks": [], "tags": []} for r in rows}
        if not tasks:
            return []
        for s in conn.execute("SELECT * FROM subtasks ORDER BY position, id"):
            if s["task_id"] in tasks:
                tasks[s["task_id"]]["subtasks"].append(
                    {"id": s["id"], "title": s["title"], "completed": bool(s["completed"])})
        for link in conn.execute(
                "SELECT tt.task_id, t.id, t.name, t.color FROM task_tags tt JOIN tags t ON t.id = tt.tag_id "
                "ORDER BY t.name"):
            if link["task_id"] in tasks:
                tasks[link["task_id"]]["tags"].append({"id": link["id"], "name": link["name"], "color": link["color"]})
        return list(tasks.values())

    def all_tasks(self) -> list[dict]:
        with self.db.connect() as conn:
            return self._load_tasks(conn)

    def get_task(self, task_id: int, conn=None) -> dict:
        if conn is None:
            with self.db.connect() as c:
                return self.get_task(task_id, c)
        found = self._load_tasks(conn, "WHERE id = ?", (task_id,))
        if not found:
            raise NotFoundError(f"Task {task_id} not found")
        return decorate(found[0], self.today())

    def _row(self, conn, task_id: int) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"Task {task_id} not found")
        return row

    # ---------------------------------------------------------------- create
    def _resolve_tag_names(self, conn, names: list[str]) -> list[int]:
        ids = []
        for raw in names:
            name = normalize_tag_name(raw)
            row = conn.execute("SELECT id FROM tags WHERE name=?", (name,)).fetchone()
            if row is None:
                cur = conn.execute("INSERT INTO tags (name, color) VALUES (?,?)", (name, _default_tag_color(name)))
                ids.append(cur.lastrowid)
            else:
                ids.append(row["id"])
        return list(dict.fromkeys(ids))

    def create_task(self, data: dict) -> dict:
        data = dict(data)
        subtasks = data.pop("subtasks", [])
        tag_ids = data.pop("tag_ids", [])
        tag_names = data.pop("tag_names", [])
        with self.db.transaction() as conn:
            fields = self._clean_fields(conn, data, partial=False)
            subtasks = self._clean_subtasks(subtasks)
            tag_ids = self._clean_tag_ids(conn, tag_ids)
            if not isinstance(tag_names, list):
                raise ValidationError("Tag names must be a list")
            tag_ids = list(dict.fromkeys(tag_ids + self._resolve_tag_names(conn, tag_names)))
            now = self._now()
            status = fields.get("status", "todo")
            max_order = conn.execute("SELECT COALESCE(MAX(order_index), 0) FROM tasks").fetchone()[0]
            record = {
                "description": "", "priority": "medium", "status": status, "category_id": None,
                "due_date": None, "estimated_minutes": None, "time_spent_minutes": 0, "pinned": 0,
                **fields,
                "order_index": max_order + 1000, "created_at": now, "updated_at": now,
                "completed_at": self._completion_after(None, None, status),
            }
            cols = ", ".join(record)
            cur = conn.execute(f"INSERT INTO tasks ({cols}) VALUES ({', '.join('?' * len(record))})",
                               tuple(record.values()))
            task_id = cur.lastrowid
            self._write_children(conn, task_id, subtasks, tag_ids)
            self._log(conn, task_id, record["title"], "created",
                      f"priority {record['priority']}, status {status}", now)
            return self.get_task(task_id, conn)

    def create_from_capture(self, text: str, *, category_id: int | None = None, status: str | None = None,
                            default_due: str | None = None) -> dict:
        """Create a task from a quick-capture line.

        ``default_due`` (e.g. the clicked calendar day) applies only when the
        line contains no recognized due-date token.
        """
        if not isinstance(text, str):
            raise ValidationError("Capture text is required")
        parsed = parse_capture(text, self.today())
        if not parsed.title:
            raise ValidationError("Title is required")
        data = {"title": parsed.title, "priority": parsed.priority, "tag_names": parsed.tags}
        if parsed.estimated_minutes is not None:
            data["estimated_minutes"] = min(parsed.estimated_minutes, MAX_MINUTES)
        due = parsed.due_date.isoformat() if parsed.due_date else default_due
        if due:
            data["due_date"] = due
        if category_id is not None:
            data["category_id"] = category_id
        if status is not None:
            data["status"] = status
        return self.create_task(data)

    @staticmethod
    def _write_children(conn, task_id: int, subtasks: list[dict], tag_ids: list[int]) -> None:
        conn.execute("DELETE FROM subtasks WHERE task_id=?", (task_id,))
        conn.executemany(
            "INSERT INTO subtasks (task_id, title, completed, position) VALUES (?,?,?,?)",
            [(task_id, s["title"], int(s["completed"]), i) for i, s in enumerate(subtasks)])
        conn.execute("DELETE FROM task_tags WHERE task_id=?", (task_id,))
        conn.executemany("INSERT INTO task_tags (task_id, tag_id) VALUES (?,?)",
                         [(task_id, tid) for tid in tag_ids])

    # ---------------------------------------------------------------- update
    def _apply(self, conn, row: sqlite3.Row, fields: dict, action: str | None = None, details: str = "",
               extra_changes: tuple[str, ...] = ()) -> None:
        now = self._now()
        updates = dict(fields)
        if "status" in fields:
            updates["completed_at"] = self._completion_after(row["status"], row["completed_at"], fields["status"])
        updates["updated_at"] = now
        assignments = ", ".join(f"{k}=?" for k in updates)
        conn.execute(f"UPDATE tasks SET {assignments} WHERE id=?", (*updates.values(), row["id"]))
        if action is None:
            became_done = fields.get("status") == "completed" and row["status"] != "completed"
            reopened = "status" in fields and fields["status"] != "completed" and row["status"] == "completed"
            action = "completed" if became_done else "updated"
            if not details:
                changed = sorted({k for k, v in fields.items() if row[k] != v} | set(extra_changes))
                if reopened:
                    changed = ["reopened"] + [c for c in changed if c != "status"]
                details = ", ".join(changed) or "saved with no changes"
        self._log(conn, row["id"], fields.get("title", row["title"]), action, details, now)

    def update_task(self, task_id: int, data: dict) -> dict:
        """Full update from the detail editor: replaces subtasks and tags wholesale."""
        data = dict(data)
        if "subtasks" not in data or "tag_ids" not in data:
            raise ValidationError("A full update must include subtasks and tag_ids")
        subtasks = data.pop("subtasks")
        tag_ids = data.pop("tag_ids")
        with self.db.transaction() as conn:
            row = self._row(conn, task_id)
            fields = self._clean_fields(conn, data, partial=False)
            subtasks = self._clean_subtasks(subtasks)
            tag_ids = self._clean_tag_ids(conn, tag_ids)
            old_subtasks = [(r["title"], bool(r["completed"])) for r in conn.execute(
                "SELECT title, completed FROM subtasks WHERE task_id=? ORDER BY position, id", (task_id,))]
            old_tags = {r["tag_id"] for r in conn.execute("SELECT tag_id FROM task_tags WHERE task_id=?", (task_id,))}
            extra = []
            if old_subtasks != [(s["title"], s["completed"]) for s in subtasks]:
                extra.append("subtasks")
            if old_tags != set(tag_ids):
                extra.append("tags")
            self._apply(conn, row, fields, extra_changes=tuple(extra))
            self._write_children(conn, task_id, subtasks, tag_ids)
            return self.get_task(task_id, conn)

    def patch_task(self, task_id: int, data: dict) -> dict:
        """Partial update: only supplied fields change."""
        if not isinstance(data, dict) or not data:
            raise ValidationError("No fields to update")
        with self.db.transaction() as conn:
            row = self._row(conn, task_id)
            fields = self._clean_fields(conn, data, partial=True)
            self._apply(conn, row, fields)
            return self.get_task(task_id, conn)

    def move_to_quadrant(self, task_id: int, target: str) -> dict:
        with self.db.transaction() as conn:
            row = self._row(conn, task_id)
            task = dict(row)
            if task["status"] == "completed":
                raise ValidationError("Completed tasks are not part of the matrix")
            changes = matrix_drop_changes(task, target, self.today())
            self._apply(conn, row, changes, "updated", f"moved to matrix quadrant {target}")
            return self.get_task(task_id, conn)

    def reorder(self, items: list, status: str | None = None) -> list[dict]:
        """Set ordering indexes for a batch; optionally move them all to ``status``.

        An existing completion timestamp is preserved when status stays completed.
        """
        if not isinstance(items, list) or not items:
            raise ValidationError("Nothing to reorder")
        if status is not None and status not in STATUSES:
            raise ValidationError("Invalid status")
        with self.db.transaction() as conn:
            now = self._now()
            for item in items:
                if (not isinstance(item, dict) or isinstance(item.get("id"), bool)
                        or not isinstance(item.get("id"), int) or isinstance(item.get("order_index"), bool)
                        or not isinstance(item.get("order_index"), int)):
                    raise ValidationError("Each item needs integer id and order_index")
                row = self._row(conn, item["id"])
                updates = {"order_index": item["order_index"]}
                if status is not None and status != row["status"]:
                    updates["status"] = status
                    updates["completed_at"] = self._completion_after(row["status"], row["completed_at"], status)
                    updates["updated_at"] = now
                    action = "completed" if status == "completed" else "updated"
                    self._log(conn, row["id"], row["title"], action, f"moved to {status}", now)
                conn.execute(f"UPDATE tasks SET {', '.join(f'{k}=?' for k in updates)} WHERE id=?",
                             (*updates.values(), row["id"]))
            return [self.get_task(item["id"], conn) for item in items]

    def add_focus_minutes(self, task_id: int, minutes) -> dict:
        if isinstance(minutes, bool) or not isinstance(minutes, int) or not 1 <= minutes <= 24 * 60:
            raise ValidationError("Focus minutes must be a whole number between 1 and 1440")
        with self.db.transaction() as conn:
            row = self._row(conn, task_id)
            if row["deleted"]:
                raise ValidationError("Cannot log time on a trashed task")
            now = self._now()
            conn.execute("UPDATE tasks SET time_spent_minutes = time_spent_minutes + ?, updated_at=? WHERE id=?",
                         (minutes, now, task_id))
            self._log(conn, task_id, row["title"], "updated", f"logged {minutes} focus minutes", now)
            return self.get_task(task_id, conn)

    # ------------------------------------------------------------- lifecycle
    def delete_task(self, task_id: int, permanent: bool = False) -> str:
        """Trash an active task; purge a task already in trash (or on explicit request)."""
        with self.db.transaction() as conn:
            row = self._row(conn, task_id)
            now = self._now()
            if permanent or row["deleted"]:
                self._log(conn, task_id, row["title"], "permanently deleted", "removed with subtasks and tag links", now)
                conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
                return "purged"
            conn.execute("UPDATE tasks SET deleted=1, updated_at=? WHERE id=?", (now, task_id))
            self._log(conn, task_id, row["title"], "moved to trash", "recoverable from Trash", now)
            return "trashed"

    def _set_flag(self, task_id: int, column: str, value: int, action: str, details: str) -> dict:
        with self.db.transaction() as conn:
            row = self._row(conn, task_id)
            now = self._now()
            conn.execute(f"UPDATE tasks SET {column}=?, updated_at=? WHERE id=?", (value, now, task_id))
            self._log(conn, task_id, row["title"], action, details, now)
            return self.get_task(task_id, conn)

    def restore_task(self, task_id: int) -> dict:
        return self._set_flag(task_id, "deleted", 0, "restored", "restored from Trash")

    def archive_task(self, task_id: int) -> dict:
        return self._set_flag(task_id, "archived", 1, "archived", "moved to Archive")

    def unarchive_task(self, task_id: int) -> dict:
        return self._set_flag(task_id, "archived", 0, "unarchived", "returned from Archive")

    def bulk(self, action: str, ids: list, value=None) -> int:
        if action not in BULK_ACTIONS:
            raise ValidationError("Unknown bulk action")
        if not isinstance(ids, list) or not ids or any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
            raise ValidationError("Select at least one task")
        ids = list(dict.fromkeys(ids))
        with self.db.transaction() as conn:
            if action == "category":
                fields_template = self._clean_fields(conn, {"category_id": value}, partial=True)
            elif action == "priority":
                fields_template = self._clean_fields(conn, {"priority": value}, partial=True)
            now = self._now()
            for task_id in ids:
                row = self._row(conn, task_id)
                if action in ("complete", "uncomplete"):
                    fields = {"status": "completed" if action == "complete" else "todo"}
                    if row["status"] == fields["status"]:
                        continue
                    self._apply(conn, row, fields, "batch operation",
                                "marked complete" if action == "complete" else "marked incomplete")
                elif action in ("priority", "category"):
                    self._apply(conn, row, fields_template, "batch operation",
                                f"set {action} to {value if value is not None else 'none'}")
                elif action == "trash":
                    conn.execute("UPDATE tasks SET deleted=1, updated_at=? WHERE id=?", (now, task_id))
                    self._log(conn, task_id, row["title"], "batch operation", "moved to trash", now)
                elif action == "restore":
                    conn.execute("UPDATE tasks SET deleted=0, updated_at=? WHERE id=?", (now, task_id))
                    self._log(conn, task_id, row["title"], "batch operation", "restored from trash", now)
                elif action == "purge":
                    self._log(conn, task_id, row["title"], "batch operation", "permanently deleted", now)
                    conn.execute("DELETE FROM tasks WHERE id=?", (task_id,))
                elif action in ("archive", "unarchive"):
                    conn.execute("UPDATE tasks SET archived=?, updated_at=? WHERE id=?",
                                 (int(action == "archive"), now, task_id))
                    self._log(conn, task_id, row["title"], "batch operation", f"{action}d", now)
            return len(ids)

    # ------------------------------------------------------------ categories
    def list_categories(self) -> list[dict]:
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute("SELECT * FROM categories ORDER BY id")]

    def create_category(self, name: str, color: str | None = None, icon: str = "folder") -> dict:
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Category name is required")
        if len(name.strip()) > 60:
            raise ValidationError("Category name is too long")
        if color is None:
            color = self.rng.choice(CATEGORY_COLORS)
        if not _is_color(color):
            raise ValidationError("Invalid color")
        if not isinstance(icon, str) or not icon:
            icon = "folder"
        with self.db.transaction() as conn:
            cur = conn.execute("INSERT INTO categories (name, icon, color) VALUES (?,?,?)",
                               (name.strip(), icon, color))
            return dict(conn.execute("SELECT * FROM categories WHERE id=?", (cur.lastrowid,)).fetchone())

    def delete_category(self, category_id: int) -> None:
        """Detach tasks (category set to none); tasks themselves are kept."""
        with self.db.transaction() as conn:
            if conn.execute("DELETE FROM categories WHERE id=?", (category_id,)).rowcount == 0:
                raise NotFoundError("Category not found")

    # ------------------------------------------------------------------ tags
    def list_tags(self) -> list[dict]:
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT t.id, t.name, t.color, "
                "(SELECT COUNT(*) FROM task_tags tt JOIN tasks k ON k.id = tt.task_id "
                " WHERE tt.tag_id = t.id AND k.deleted = 0) AS usage "
                "FROM tags t ORDER BY t.name")]

    def create_tag(self, name: str, color: str | None = None) -> tuple[dict, bool]:
        """Returns (tag, created). An existing name returns the existing tag."""
        name = normalize_tag_name(name)
        if color is not None and not _is_color(color):
            raise ValidationError("Invalid color")
        with self.db.transaction() as conn:
            row = conn.execute("SELECT id, name, color FROM tags WHERE name=?", (name,)).fetchone()
            if row is not None:
                return dict(row), False
            cur = conn.execute("INSERT INTO tags (name, color) VALUES (?,?)",
                               (name, color or _default_tag_color(name)))
            return {"id": cur.lastrowid, "name": name, "color": color or _default_tag_color(name)}, True

    def delete_tag(self, tag_id: int) -> None:
        """Remove the tag and its task links; the tasks are never deleted."""
        with self.db.transaction() as conn:
            if conn.execute("DELETE FROM tags WHERE id=?", (tag_id,)).rowcount == 0:
                raise NotFoundError("Tag not found")

    # -------------------------------------------------------------- activity
    def activity(self, limit: int = 50) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValidationError("Limit must be between 1 and 1000")
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT * FROM activity_log ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))]

    def clear_activity(self) -> int:
        with self.db.transaction() as conn:
            return conn.execute("DELETE FROM activity_log").rowcount

    # ------------------------------------------------------------------ seed
    def seed_if_first_run(self) -> bool:
        """Insert demonstration content once. A ``seeded`` flag prevents re-seeding
        after the user deletes everything. Timestamps are the real seeding time."""
        with self.db.connect() as conn:
            if conn.execute("SELECT 1 FROM meta WHERE key='seeded'").fetchone():
                return False
            has_data = conn.execute("SELECT EXISTS(SELECT 1 FROM tasks) OR EXISTS(SELECT 1 FROM categories)").fetchone()[0]
        if not has_data:
            self._seed()
        with self.db.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('seeded', ?)", (self._now(),))
        return not has_data

    def mark_seeded(self) -> None:
        with self.db.transaction() as conn:
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('seeded', ?)", (self._now(),))

    def _seed(self) -> None:
        cats = {}
        for name, icon, color in (("Work & Projects", "briefcase", "#6366f1"),
                                  ("Personal & Life", "home", "#10b981"),
                                  ("Dev & Engineering", "code", "#06b6d4"),
                                  ("Health & Fitness", "heart", "#f43f5e"),
                                  ("Finance & Bills", "wallet", "#f59e0b")):
            cats[name] = self.create_category(name, color, icon)["id"]
        tags = {}
        for name, color in (("frontend", "#06b6d4"), ("backend", "#8b5cf6"), ("design", "#ec4899"),
                            ("critical", "#f43f5e"), ("learning", "#10b981")):
            tags[name] = self.create_tag(name, color)[0]["id"]
        today = self.today()
        rel = lambda days: (today + timedelta(days=days)).isoformat()  # noqa: E731
        demo = "Demo task created on first run. "
        self.create_task({
            "title": "Welcome to Zenith — try the quick capture bar",
            "description": demo + "Type e.g. 'Review PR !high #backend ~30m @tomorrow' and watch the chips.",
            "priority": "high", "category_id": cats["Work & Projects"], "due_date": rel(0), "pinned": True,
            "estimated_minutes": 15,
            "subtasks": [{"title": "Add a task with tokens"}, {"title": "Open the board view"},
                         {"title": "Drag a task in the matrix"}],
            "tag_ids": [tags["learning"]]})
        self.create_task({
            "title": "Fix login redirect bug", "description": demo + "Users land on a blank page after sign-in.",
            "priority": "urgent", "status": "in_progress", "category_id": cats["Dev & Engineering"],
            "due_date": rel(0), "pinned": True, "estimated_minutes": 90,
            "subtasks": [{"title": "Reproduce locally", "completed": True}, {"title": "Write regression test"},
                         {"title": "Ship fix"}],
            "tag_ids": [tags["backend"], tags["critical"]]})
        self.create_task({
            "title": "Design settings page mockups", "description": demo,
            "priority": "medium", "status": "review", "category_id": cats["Work & Projects"],
            "due_date": rel(1), "estimated_minutes": 120,
            "subtasks": [{"title": "Wireframes", "completed": True}, {"title": "High-fidelity pass"}],
            "tag_ids": [tags["design"], tags["frontend"]]})
        self.create_task({
            "title": "Pay electricity bill", "description": demo,
            "priority": "low", "category_id": cats["Finance & Bills"], "due_date": rel(7),
            "estimated_minutes": 10, "tag_ids": []})
        self.create_task({
            "title": "Morning run 5 km", "description": demo,
            "priority": "medium", "status": "completed", "category_id": cats["Health & Fitness"],
            "estimated_minutes": 30,
            "subtasks": [{"title": "Stretch", "completed": True}, {"title": "Run", "completed": True}],
            "tag_ids": []})
