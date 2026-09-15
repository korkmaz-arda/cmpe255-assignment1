# Zenith Dynamic Task Workspace

A single-user productivity workspace for managing tasks. A local Python server owns a SQLite task store. A browser client shows the same tasks in five views: a date-bucketed **list**, a **Kanban board**, an **Eisenhower matrix**, a **monthly calendar**, and an **analytics** dashboard. Tasks are captured from one line of text that a deterministic, rule-based tokenizer parses (no model, no external service). Every open tab stays in sync live.

This is a productivity app, not a machine-learning project. The only analytical part is *descriptive* telemetry (completion rate, streak, a heuristic productivity index, a 30-day activity heatmap, per-priority and per-category breakdowns). It is recomputed from the database on every request, and nothing is trained, sampled, or cached.

## Quick start

```bash
conda activate cmpe255
pip install -r requirements.txt

# run the app (creates data/zenith.db and seeds demo content on first run)
python -m app                 # http://127.0.0.1:8000
python -m app --no-seed       # start with an empty store instead
python -m app --port 9000 --db /path/to/other.db

# fast tests (unit + API; offline, ~1 s)
pytest

# browser end-to-end tests (one-time browser download into ~/.cache/ms-playwright)
python -m playwright install chromium
pytest -m e2e
```

There is no dataset to download and nothing to train. Delete `data/zenith.db*` to start over.

`--host 0.0.0.0` makes the app reachable from other machines on your network. **The app has no authentication**, so only do this on a network you trust.

## Architecture

```
zenith/                 framework-independent core (no web imports)
  clock.py              injectable "today" (system clock in production, fixed clock in tests)
  db.py                 SQLite schema: CHECK-constrained enums, cascades, WAL, foreign keys on every connection
  store.py              every mutation + validation, completion-timestamp lifecycle, trash/archive, tags, audit log, seed
  capture.py            quick-capture tokenizer
  views.py              derived data: date buckets, Eisenhower quadrants + drop rule, filter/sort, sidebar counts, month grid
  analytics.py          telemetry
app/
  server.py             thin Starlette JSON routes + Server-Sent Events stream
  events.py             in-process broadcast hub
  __main__.py           `python -m app`
  static/               index.html, styles.css, js/ (plain ES modules, no build step, no third-party JS)
tests/                  pytest suites for the core and the HTTP layer
tests/e2e/              Playwright browser tests (marker `e2e`, excluded from plain `pytest`)
```

**Python is the source of truth.** It holds the task data, validation, persistence, and every rule: "today", overdue status, weekday resolution, buckets, quadrant membership, sort order, counts, and telemetry. The browser JavaScript renders what the server returns and turns gestures (drag-and-drop, keyboard, clicks) into API calls. The quick-capture preview calls `POST /api/parse`, so the tokenizer exists only in Python.

**Live sync.** Every mutation publishes a typed event (`task_created`, `task_updated`, `task_trashed`, `task_deleted`, `tasks_reordered`, `batch_completed`, `batch_updated`, `categories_updated`, `tags_updated`, `activity_cleared`) over `GET /api/events`. Each client reacts the same way: it refetches tasks, sidebar counts, categories, tags, analytics, and the calendar. A `connected` message is sent when a client attaches, and the browser's `EventSource` reconnects automatically. The only optimistic update is completion toggling. The server response replaces it, and a failed request rolls it back with an error toast.

## Features and where they live

| ID | Behavior | Implementation |
|---|---|---|
| F01 | Persistent tasks, subtasks, tags, categories; demo seed on first run only | `zenith/db.py`, `store.py` (`seeded` flag in `meta`) |
| F02 | Quick capture with live chip preview; category selector; inherits the active category filter | `capture.py`, `/api/parse`, `app.js` |
| F03 | Five views sharing filter, search, and sort | view switcher in `app.js` |
| F04 | List with six collapsible date buckets and rich rows | `views.bucket`, `views.js` |
| F05 | Board: drag between columns changes status; per-column quick-add | `store.reorder`, `views.js` |
| F06 | Matrix derived from priority + due date; drops rewrite exactly those fields | `views.quadrant`, `views.matrix_drop_changes` |
| F07 | Month calendar (Sunday start), prev/next/today, click a day to schedule | `views.month_grid`, `views.js` |
| F08 | Analytics recomputed per request | `analytics.py` |
| F09 | Detail editor; saving replaces subtasks and tags wholesale; Esc closes | `modals.js`, `store.update_task` |
| F10 | Completion toggle with chime and confetti, optimistic update | `app.js`, `feedback.js` |
| F11 | Live multi-tab and multi-client sync | `events.py`, `/api/events` |
| F12 | Smart lists, categories, tag filter, Archive, Trash, search, six sorts, pinned first | `views.filter_tasks/sort_tasks/sidebar_counts` |
| F13 | Audit log of every task mutation, with viewer and clear | `store.activity`, Activity log modal |
| F14 | Pomodoro (25/5/15) that logs focused time to a bound task | `modals.js`, `/api/tasks/{id}/focus` |
| S01 | Multi-select with bulk complete/reopen/priority/category/archive/trash (restore and delete forever in Trash) | bulk bar, `store.bulk` |
| S02 | Command palette (Ctrl/⌘ K): views, theme, accents, tools, up to 5 tasks, categories | `modals.js` |
| S03 | Shortcuts `Ctrl/⌘ K`, `/`, `?`, `Esc` (suppressed while typing) plus a reference sheet | `app.js`, `modals.js` |
| S04 | Inline category creation (random palette color, folder icon), delete | sidebar |
| S05 | Tag manager: 10-color palette, usage counts, normalization, duplicates reuse | `modals.js`, `store.create_tag` |
| S06 | Dark/light and five accents, stored in `localStorage` | `app.js`, `styles.css` |
| S07 | Web Audio cues (complete, uncomplete, click, fanfare, delete); mute stored in `localStorage` | `feedback.js` |
| S08 | Pin from a row or the editor; pinned tasks always sort first | `views.sort_tasks` |
| S09 | Delete sends an active task to Trash and permanently removes a trashed one; restore; archive | `store.delete_task` etc. |
| S10 | Collapsible sidebar (icon rail) | `app.js`, `styles.css` |
| S11 | Loading and empty states (analytics, list, quadrants, palette, trash) | `views.js`, `modals.js` |

### Quick-capture syntax

| Token | Meaning |
|---|---|
| `!urgent` `!crit` `!critical` `!u` · `!high` `!h` `!important` · `!medium` `!med` `!m` `!normal` · `!low` `!l` | priority (first `!word`; unknown words leave `medium`) |
| `#name` | tag (every occurrence; lowercased, de-duplicated; created if new) |
| `~45`, `~30m`, `~1.5h` | estimate in minutes (`h`/`hr` × 60, rounded) |
| `@today` `@tod` `@tomorrow` `@tom` `@yesterday` `@nextweek` `@2026-10-01` `@fri` `@friday` | due date (weekday = next occurrence after today) |

A token must start a word, so `bob@example.com` and `C#` are left alone, and trailing punctuation is not part of a token (`@friday.` still sets Friday; the punctuation stays in the title). Only the first priority, estimate, and due-date token is used. A recognized token kind is removed from the title even when its value is not understood (`@someday` gives no date but disappears). Only calendar dates are supported, not clock times.

### Telemetry definitions (F08)

All figures use active tasks only (not archived, not trashed) and the server's local date.

- **Completion velocity** = completed ÷ active tasks, rounded to a whole percent.
- **Active streak**: consecutive days, counting back from today within the last 30 days, with at least one completion. No completion *today* does not break the streak; any earlier empty day does.
- **Productivity index** = `round(velocity × 0.7) + min(streak × 5, 20) − min(overdue × 10, 30) + 20`, clamped to 10–100. This is an ad-hoc heuristic score, not a validated measure, and the UI labels it that way.
- **Focus time logged** = sum of time-spent minutes (hours shown to one decimal).
- **30-day heatmap**: one cell per day, oldest first, with completions and creations. Shade levels: 4 = ≥4 completions, 3 = ≥2, 2 = ≥1, 1 = tasks created but none completed, 0 = no activity.
- **Completion by priority**: bar length = completed ÷ total for that priority, matching its label.
- **Tasks by category**: faint bar = total tasks, solid bar = completed tasks, both on the same scale. Tasks with no category are grouped as "Uncategorized".

Completion history comes from each task's current `completed_at`, so reopening a task removes that day's completion from the streak and heatmap.

## Deliberate fixes to the source behavior

The specification documents several defects in the original project. This implementation keeps the intended behavior instead:

- **Capture tags attach.** Tag names from quick capture are resolved to existing tags or created.
- **Severity sort.** "Priority" sorts urgent → high → medium → low, not alphabetically.
- **Sidebar counts describe their destination** (whole store), not the list you are currently viewing.
- **Matrix drops always land in the target quadrant.** Rule for date-urgent tasks (due today or overdue):
  - **Do First**: priority urgent, due today.
  - **Delegate**: priority medium, due today.
  - **Schedule**: priority high; the due date moves to tomorrow.
  - **Eliminate / Backlog**: priority low; the due date is cleared.
  - Non-urgent due dates are left untouched, and no other field changes.
- **Priority bar matches its label** (both show completion %).
- **Heatmap excludes trashed and archived tasks**, like every other aggregate.
- **Pomodoro logs time actually focused.** A completed session logs its measured duration. Resetting, switching mode, re-binding, or closing mid-session logs the partial time (whole minutes, rounded; under 1 minute logs nothing). Paused time is not counted.
- **Server features you can now reach from the UI:** restore, delete forever, archive/unarchive, bulk reopen, category and tag deletion, and an Activity log viewer.
- **Calendar scheduling** uses an inline composer that accepts capture tokens. The clicked day is the default due date, and an explicit `@date` token overrides it.
- **Board quick-add into Done** sets the completion timestamp.

**Completion timestamp lifecycle:**
- not done → done stamps the current time
- done → done keeps the existing stamp
- reopening clears it
- completing again stamps a new time
- reorders and unrelated edits never change it

**Deleting a category** keeps its tasks and leaves them uncategorized. **Deleting a tag** removes it from its tasks but never deletes the tasks.

## Demo content

On the first run against an empty database, the app creates five categories, five tags, and five sample tasks. Their descriptions say "Demo task created on first run". They use real timestamps from the moment of seeding (nothing is backdated). The one pre-completed sample therefore counts as a completion *today* in analytics. The seed never runs again, even after you delete everything. Use `--no-seed` for a clean start.

## Tests

- **`pytest`** (fast, offline, fixed clock, temporary DB) covers:
  - every tokenizer rule
  - storage constraints and foreign keys
  - persistence across reopen, and one-time seeding
  - ordering (+1000 per new task)
  - the completion-timestamp lifecycle
  - full vs partial updates
  - trash/purge/restore/archive
  - category and tag deletion semantics
  - tag normalization and reuse
  - bulk actions and the audit log (details list only the fields that actually changed)
  - buckets, and every quadrant drop (all priorities × past/today/future/no date) landing in its target with no unrelated changes
  - severity sort and pinned-first under every sort
  - filter composition and view-independent counts
  - month grid
  - telemetry formulas and edge cases
  - HTTP routes, validation errors, and event publication
- **`pytest -m e2e`** (Playwright + Chromium, a real server per test) covers:
  - capture chips and creation
  - board drag and column quick-add
  - matrix drops
  - keyboard shortcuts, including suppression while typing
  - command-palette arrow/Enter navigation
  - two-tab live sync
  - trash → restore → delete forever
  - optimistic completion (held request and failure rollback)
  - sound cue events, confetti, and persisted mute
  - the Pomodoro timer (partial, paused, full, break), driven by Playwright's test clock
  - calendar scheduling (default date and override)
  - editor subtasks and tags
  - analytics rendering
  - bulk actions and sidebar collapse

## Limitations

- Single user, no authentication. "Today" is the server machine's local date, so clients in other time zones follow the server.
- Drag-and-drop uses HTML5 drag events, which work with a mouse. Touch-only devices can't drag, but can still change status and priority from the detail editor.
- The Pomodoro countdown lives in the open dialog. Closing the dialog logs any partial focus time and resets it, as specified.
- Live sync is in-process. Run a single server process (the default) so every client shares one event hub.
