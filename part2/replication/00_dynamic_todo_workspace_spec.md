# Project Specification — Dynamic Task & Workflow Workspace (`00_dynamic_todo_workspace`)

Source project inspected statically and read-only. Branded in the source as "Zenith Task".

---

## 1. Project Overview

A single-user, full-stack productivity workspace for managing tasks. A persistent server owns a relational task store and exposes it to a rich browser client that renders the same task set through five interchangeable views (smart list, Kanban board, Eisenhower matrix, monthly calendar, analytics dashboard). Tasks are captured through a single free-text field whose contents are parsed into structured attributes by a deterministic, rule-based tokenizer (no model inference, no external service).

The project is not a machine-learning project despite the surrounding repository framing. Its only analytical identity is descriptive productivity telemetry computed on demand from the task store: completion rates, a consecutive-completion streak, a composite productivity index, a 30-day activity heatmap, and priority/category breakdowns. All quantities are simple aggregations over stored rows, recomputed on every request; nothing is trained, fitted, sampled, or persisted as a model artifact.

Primary user experience: a keyboard-friendly, glassmorphic dark-first workspace with instant feedback (synthesized audio cues, confetti on completion), live multi-tab synchronization, and a Pomodoro focus timer that writes measured time back onto a task.

---

## 2. Required Features

### Core Requirements

**F01 — Persistent relational task workspace**

Behavior: Tasks and their supporting entities persist across restarts in a server-side relational store.

Important details:
- Task attributes: title (required, trimmed), description, priority (`low` | `medium` | `high` | `urgent`, default `medium`), workflow status (`todo` | `in_progress` | `review` | `completed`, default `todo`), optional category reference, optional due date (calendar date, no time component), estimated minutes, time-spent minutes, numeric ordering index, pinned flag, archived flag, soft-deleted flag, creation/update timestamps, and a completion timestamp.
- Related entities: ordered subtasks (title + completed flag) owned by a task and removed with it; a shared tag vocabulary (name + color) associated many-to-many with tasks; categories (name, icon identifier, color).
- Priority and status values are constrained to their enumerations at the storage layer.
- On an empty store the application seeds demonstration content: five categories (Work & Projects, Personal & Life, Dev & Engineering, Health & Fitness, Finance & Bills), five tags (frontend, backend, design, critical, learning), and five sample tasks with subtasks, tags, relative due dates (today / tomorrow / +7 days), mixed priorities and statuses, two pinned, one already completed. Seeding is a first-run action only.

**F02 — Natural-language quick capture with live token preview**

Behavior: The user types one free-text line; the application extracts structured attributes from prefix-marked tokens, removes them from the text, and uses the remainder as the title. Recognized tokens are shown as labeled chips beneath the input as the user types, before submission.

Important details (single pass, deterministic, no network call):
- Priority: first `!word` token. Accepted synonyms → `urgent` (`urgent`, `crit`, `critical`, `u`), `high` (`high`, `h`, `important`), `medium` (`medium`, `med`, `m`, `normal`), `low` (`low`, `l`). Unrecognized words leave the default `medium`.
- Tags: every `#name` token (alphanumeric, underscore, hyphen), lowercased, de-duplicated.
- Time estimate: first `~number[unit]` token; unit `h`/`hr` multiplies by 60, otherwise minutes; result rounded to whole minutes. Fractional values are supported (`~1.5h` → 90).
- Due date: first `@token`. Supported: `today`/`tod`, `tomorrow`/`tom`, `yesterday`, `nextweek`, an explicit ISO calendar date, and weekday names or three-letter abbreviations resolved to the *next* occurrence of that weekday. Unrecognized values yield no due date but the token is still stripped from the title.
- The submit control stays disabled while the residual title is empty.
- A category selector sits beside the input; when no category is chosen and the active navigation filter is a category, the new task inherits that category.

**F03 — Five interchangeable workspace views**

Behavior: A persistent view switcher swaps the main pane between list, board, matrix, calendar, and analytics presentations of the same working task set. The active navigation filter, search term, and sort selection remain in effect across views.

**F04 — Smart date-bucketed list view**

Behavior: Tasks are grouped into collapsible, color-coded time buckets with per-bucket counts; empty buckets are hidden.

Important details:
- Buckets in order: Overdue, Due Today, Due Tomorrow, Upcoming & Scheduled, No Due Date, Completed. Completed tasks go to the Completed bucket regardless of date.
- Each row shows: multi-select box, completion checkbox, title, pin marker, due-date chip (label "Today" or short month/day; overdue styling with a warning icon), time chip (`spent / estimate` when time has been logged), category chip in category color, subtask progress as `done/total` plus a miniature progress bar, and tag chips.
- Hover actions per row: pin/unpin, open detail editor, delete.
- An explicit empty state appears when the current filter yields no tasks.

**F05 — Kanban board with drag-driven status changes**

Behavior: Four fixed columns — To Do, In Progress, In Review, Done — each with a live count. Dragging a card to another column changes the task's workflow status; dropping into Done triggers the completion feedback. Each column offers an inline quick-add that creates a task directly in that column's status.

Important details: cards show priority label, category, title, truncated description, due date, subtask ratio, and tags; clicking a card opens the detail editor.

**F06 — Eisenhower decision matrix with derived classification**

Behavior: Incomplete tasks are placed into four quadrants derived from the task's own attributes rather than a stored quadrant field. Dragging a task into a quadrant rewrites the attributes that define that quadrant.

Important details:
- Urgency is derived: a task is urgent if its priority is `urgent`, or its due date is today or in the past.
- Importance is derived: priority is `urgent` or `high`.
- Quadrants: Do First (urgent + important), Schedule (important, not urgent), Delegate (urgent, not important), Eliminate/Backlog (neither). Each shows a title, explanatory subtitle, count, and an empty-state hint.
- Drop effects: Do First → priority `urgent` and due date set to today; Schedule → priority `high`; Delegate → priority `medium` and due date set to today; Eliminate → priority `low`.
- Completed tasks are excluded from all quadrants.

**F07 — Monthly calendar timeline**

Behavior: A month grid (weeks starting Sunday, leading/trailing days dimmed) places each task on its due date as a chip colored by priority, struck through when completed, with a per-day task count. Navigation: previous month, next month, jump to today. Clicking an empty area of a day prompts for a title and creates a task scheduled on that day; clicking a chip opens the detail editor.

**F08 — Productivity analytics dashboard**

Behavior: A telemetry view computed server-side on every request from the current task store (never cached, never precomputed).

Important details — headline metrics:
- Completion velocity: completed ÷ total non-deleted tasks, as a rounded percentage, with the underlying counts.
- Active streak: consecutive days, counted backwards from today within the 30-day window, having at least one completion. A zero-completion *today* does not break the streak (the count then ends yesterday); any earlier zero day ends it.
- Productivity index (0–100): `round(completionRate × 0.7) + min(streak × 5, 20) − min(overdue × 10, 30) + 20`, clamped to the range 10–100. This is an ad-hoc composite, not a validated metric; it is labeled to the user as derived from velocity and deadline compliance.
- Focus time logged: sum of time-spent minutes, displayed in hours to one decimal alongside the raw minute total.

Other panels:
- 30-day activity heatmap: one cell per day for the trailing 30 days (oldest first), each carrying completions and creations for that day; intensity has four levels driven by completion count (≥4, ≥2, ≥1) with total-activity fallbacks; hovering reveals the date and both counts; a Less→More legend is shown.
- Priority distribution: one row per priority level (urgent, high, medium, low) showing completed/total and completion percentage, with a bar.
- Category breakdown: per category (uncategorized tasks grouped under a placeholder name and color) the completed and total task counts.
- Additional summary counts are produced and available: per-status counts and an overdue count (incomplete tasks with a past due date).
- Deleted tasks are excluded from the summary and breakdown aggregations.

**F09 — Task detail editor**

Behavior: A modal editor over a single task exposing every editable property and saving them in one action.

Important details: title, description, status, priority, category, due date, estimated minutes, time-spent minutes, pin toggle, delete, and the task's identifier for reference. A subtask checklist supports adding, toggling, and removing items with a completed/total counter; ordering follows list position. Tags are assigned by toggling chips from the shared tag vocabulary. Saving replaces the task's subtask list and tag set wholesale with the edited state. Cancel/close discards; Escape also closes.

**F10 — Completion toggling with immediate feedback**

Behavior: Toggling a task's checkbox moves it between completed and to-do, stamping or clearing the completion timestamp. Completing plays an ascending four-note chime and fires a confetti burst; un-completing plays a short descending tone. The row updates optimistically before the server confirms.

**F11 — Live synchronization across open clients**

Behavior: Any task, category, or tag mutation is broadcast to all connected clients, which refresh their working set so every open window (including other tabs and other machines) converges without manual reload. A connection handshake message is delivered when a client attaches, and the transport reconnects automatically on drop.

Important details: broadcasts are typed (task created, task updated, task trashed, task permanently deleted, tasks reordered, batch completed, categories updated, tags updated). The client's reaction is uniform: a full refetch of tasks, categories, tags, and analytics. Preserving *that* convergence behavior matters; the specific delivery mechanism does not.

**F12 — Navigation, filtering, and search**

Behavior: A persistent sidebar drives the working set.

Important details:
- Smart lists: All Tasks, Today (due today), Upcoming (due after tomorrow), Important (priority urgent or high), Completed — each with a count badge.
- Category list with per-category task counts (deleted tasks excluded) and a tag chip row acting as a single-tag filter toggle.
- Archive and Trash destinations, which switch the working set to archived-only or deleted-only tasks respectively.
- A global search box matches the term against task titles and descriptions (case-insensitive substring), combining with the active filter.
- Sorting choices: custom order, due date ascending/descending, priority, title A–Z, recently added. Pinned tasks always sort ahead of unpinned ones regardless of the chosen sort.

**F13 — Activity audit trail**

Behavior: Every task mutation records a dated entry capturing the task identifier, the task title at the time, an action label (created, updated, completed, deleted, archived, moved to trash, permanently deleted, batch operation), and a short detail string. The log is retrievable newest-first with a configurable limit and can be cleared wholesale.

Note: the log is written and readable but no view in the shipped interface renders it.

**F14 — Pomodoro focus timer with time attribution**

Behavior: A modal timer with three modes — Focus 25 minutes, Short Break 5 minutes, Long Break 15 minutes — offering start/pause and reset, a large monospace countdown, and a progress bar. Switching modes resets the countdown and stops the timer.

Important details: the user may bind the session to any incomplete task. When a focus session reaches zero, a fanfare plays, confetti fires, a completed-session counter increments, an acknowledgment message is shown, and 25 minutes are added to the bound task's time-spent total (a fixed 25, independent of the mode's actual configured duration). Break completion only notifies.

### Secondary Requirements

**S01 — Multi-select and bulk actions.** Rows carry selection boxes and the list offers select-all/deselect-all. With a selection active a floating action bar shows the count and offers: mark all complete, set priority for all, set category for all, move all to trash, and clear selection. The underlying store additionally supports bulk un-complete, restore from trash, permanent delete, archive, and unarchive.

**S02 — Command palette.** A searchable overlay (opened by shortcut or toolbar button) listing view switches, theme toggle, and accent presets; typing also surfaces up to five matching tasks (opening the detail editor) and matching categories (applying the category filter). Arrow keys move the highlight, Enter runs the highlighted entry, Escape dismisses, and an explicit no-results state is shown.

**S03 — Keyboard shortcuts and cheat sheet.** Global bindings: Ctrl/Cmd+K toggles the command palette, `/` focuses global search, `?` toggles the shortcuts reference, Escape closes any open overlay. The `/` and `?` bindings are suppressed while typing in a field. A reference modal lists these plus the quick-capture token syntax.

**S04 — Category management.** Inline creation of a category from the sidebar (name entered, a color chosen at random from a fixed palette, generic folder icon). Category icons render from a small fixed icon set with a fallback for unknown identifiers.

**S05 — Tag management.** A modal for creating tags with a name and a color chosen from a ten-swatch palette, and for reviewing all existing tags with their usage counts. Tag names are normalized to lowercase with any leading `#` stripped; creating a name that already exists returns the existing tag rather than failing.

**S06 — Theming.** Dark/light mode toggle and five accent presets (indigo, cyan, emerald, amber, rose) applied as document-level attributes. Both selections persist per browser across sessions; dark and indigo are the defaults.

**S07 — Synthesized audio feedback.** Distinct generated tones for completion (ascending arpeggio), un-completion (short downward sweep), generic clicks, focus-session completion (four-note fanfare), and deletion (falling filtered tone). A mute toggle in the sidebar persists per browser, and muting suppresses all cues.

**S08 — Pin to top.** Tasks can be pinned from a row action or the detail editor; pinned tasks are marked and always ordered first.

**S09 — Trash, restore, and archive lifecycle.** Deleting an active task moves it to trash (recoverable); deleting a task already in trash, or deleting with an explicit permanent request, removes it and its subtasks/tag links irreversibly. Archived and trashed tasks are excluded from all normal views and from analytics.

**S10 — Collapsible sidebar.** The sidebar collapses to an icon rail and expands again; labels, counts, and the tag section hide when collapsed.

**S11 — Loading and empty states.** The analytics view shows a loading message until telemetry arrives; the list view shows a guidance empty state; matrix quadrants and the command palette show their own empty states.

---

## 3. User Workflow

1. The workspace opens on the list view with the seeded or persisted task set, sidebar counts, and telemetry loaded. `[F01, F03, F04, F12]`
2. The user types one line into quick capture, watches the recognized attribute chips appear, and submits. `[F02]`
3. The task appears in its date bucket; the user refines it in the detail editor, adds subtasks, and assigns tags. `[F04, F09]`
4. Work is progressed by dragging across board columns, re-triaging in the matrix, or scheduling on the calendar. `[F05, F06, F07]`
5. The user starts a focus session bound to a task; on completion the measured time lands on the task. `[F14]`
6. Completing tasks produces audio/visual feedback and updates the telemetry, streak, and heatmap. `[F10, F08]`
7. Multi-select drives bulk re-prioritization, re-categorization, completion, or trashing. `[S01]`
8. Navigation, search, sorting, the command palette, and shortcuts are used throughout to narrow the working set. `[F12, S02, S03]`
9. Any change is reflected in every other open client without a reload. `[F11]`

A second concurrent client following the same journey sees the first client's changes appear on its own screen. `[F11]`

---

## 4. Data, State, and Data-Science Behavior

**Data origin.** All data is user-generated and stored server-side. There is no external dataset, no synthetic generator, no sampling, no random seed, and no train/test partitioning anywhere in the project. The only non-user data is the first-run demonstration seed described in `F01`, whose due dates are computed relative to the seeding moment.

**Derived attributes rather than stored ones.** Several user-visible classifications are computed at render time from stored fields, not persisted:
- list date buckets (from due date and status),
- matrix urgency/importance (from priority and due date),
- overdue styling (incomplete with a past due date),
- subtask progress ratios,
- sidebar smart-list membership and counts.

**Telemetry computation (`F08`).** Pure SQL aggregation over the non-deleted task set plus post-processing in the request handler. The 30-day window is built as an explicit list of the trailing 30 calendar dates, joined against completion dates and creation dates; missing days become explicit zero entries so the heatmap always has exactly 30 cells. Streak, productivity index, and heatmap intensity levels are derived from those arrays using the rules stated in `F08`. Everything is recomputed per request; no telemetry artifact is stored.

**State and persistence model.**
- Server-owned, durable: tasks, subtasks, tags, task–tag links, categories, activity log.
- Client-owned, per-browser durable: theme, accent, sound-mute preference.
- Client-owned, session-only: active view, active filter, search term, sort selection, selection set, open modals, sidebar collapse, Pomodoro mode/countdown/session count/bound task. The Pomodoro countdown is *not* persisted and resets when the modal closes.

**Mutation semantics worth preserving.**
- Creating a task assigns it an ordering index 1000 above the current maximum, leaving gaps for reordering.
- A full task update replaces the subtask list and tag set entirely with the submitted state.
- A partial update touches only the supplied fields, and setting status to completed stamps the completion time while any other status clears it.
- Reordering accepts a batch of identifier/order pairs, optionally with a status, and preserves an existing completion timestamp when the status stays completed.
- Deleting a category detaches it from its tasks rather than deleting them; deleting a task removes its subtasks and tag links.

**Retraining / recompute.** No model exists. The recompute behavior that matters is that telemetry and the task set are re-derived from storage on every request and on every broadcast event.

---

## 5. Outputs and UI Behavior

- **Task rows and cards** communicate, at a glance: priority (left color edge and label), schedule pressure (due-date chip, overdue emphasis), effort (estimate and logged time), decomposition progress (subtask ratio and bar), classification (category chip, tag chips), and pinning.
- **Board columns** communicate throughput distribution across the four workflow stages via per-column counts.
- **Matrix quadrants** communicate triage load: how many active tasks are crises versus strategic work versus interruptions versus noise, with counts and explanatory subtitles.
- **Calendar cells** communicate deadline clustering across a month: which days are loaded, which tasks fall where, and their priority via chip color.
- **Analytics** communicates four headline numbers (completion percentage with counts, streak in days, productivity index out of 100, logged focus hours with raw minutes), a 30-day cadence picture via heatmap intensity, per-priority completion ratios, and per-category totals.
- **Feedback surfaces:** confetti and chimes on completion, a fanfare and acknowledgment on focus-session completion, hover tooltips on heatmap cells and icon buttons, keyboard hints rendered as key caps.
- **Exports:** none. The project has no download, report-generation, or print output.

---

## 6. Important Semantic Mechanisms

- **Token-stripping capture.** The title is defined as the residue after every recognized token is removed and whitespace is collapsed. Reimplementations must remove tokens from the title even when the token's value is unrecognized (e.g. an unknown `@word` still disappears from the title). Tag scanning runs against the original input while the other extractors run against the progressively stripped text.
- **Derived-not-stored Eisenhower classification.** Because quadrant membership is recomputed from priority and due date, a quadrant drop must mutate exactly those attributes; there is no quadrant column to write.
- **Broadcast-then-refetch convergence.** Clients do not merge event payloads; an event triggers a full reload of the working set, so the server's query result is always the single source of truth for what is displayed. Optimistic local updates exist only for completion toggling and are immediately superseded.
- **Server-side filtering plus client-side refinement.** Category, tag, search, archive/trash scope, and sorting are resolved by the server query; the smart lists (today, upcoming, important, completed) are applied on top of that result in the client. Both layers must exist for the sidebar and search to compose correctly.
- **Pinning overrides sort.** Pinned-first ordering is applied ahead of whatever sort the user selected.
- **Soft-delete two-stage removal.** The same delete gesture means "trash" for an active task and "purge" for a task already in trash.
- **Recompute-on-every-request telemetry.** No caching layer; analytics always reflect the store at read time.

---

## 7. Source Caveats

### Apparent implementation defects or quirks

- **Quick-capture tags are not attachable.** The capture path submits tag *names*, while the creation path expects tag *identifiers* and writes them straight into the task–tag link table with no name-to-identifier resolution. Under the store's referential constraints this cannot produce a valid link for a name that is not also an identifier. Invariant violated: a tag typed during capture should result in the task carrying that tag (creating the tag if it does not exist). *Conflicting evidence:* the committed database contains a task created during the capture era that nonetheless carries correctly resolved tag links, and the end-to-end test exercises this path using literal tag identifiers, so the failure mode could not be confirmed without execution.
- **Detail-editor hook ordering.** The detail editor returns early when no task is open and only then declares its local state. This conditional declaration violates the component model's requirement that state declarations be unconditional, and would be expected to fault when the modal transitions from closed to open. Static inspection cannot confirm the runtime outcome, but the pattern is unsafe and should not be reproduced. Invariant: opening a task's editor must render its current values reliably.
- **Priority sort is lexical.** Sorting by priority orders the stored words alphabetically (high, low, medium, urgent), so the "Priority (High to Low)" option does not produce a severity ordering. Invariant: a severity sort should follow the priority rank, not the label spelling.
- **Sidebar counts are scoped to the current query.** Counts are computed from the currently fetched task list rather than the whole store, so viewing Trash or a single category makes the other counts read as zero or as subsets. Invariant: navigation counts should describe the destination, not the current view.
- **Matrix "Schedule" and "Eliminate" drops can be no-ops.** Those quadrants only lower the priority and leave the due date untouched, so a task due today or overdue remains classified urgent and snaps back to an urgent quadrant. Invariant: dropping a task into a quadrant should place it in that quadrant.
- **Priority-distribution bar disagrees with its label.** The bar length encodes the priority level's share of all tasks while the adjacent label reports that level's completion percentage, so the two read as if they measured the same thing.
- **Heatmap completion counts ignore deletion.** Daily completions are counted across all tasks including soft-deleted ones, while daily creations and every other aggregate exclude them.
- **Fixed 25-minute time attribution.** A finished focus session always adds 25 minutes to the bound task regardless of the mode's configured duration or of a session that was paused or reset mid-way. Invariant: logged time should reflect the time actually focused.
- **Unreachable server capabilities.** Archive/unarchive, restore-from-trash, bulk un-complete, bulk permanent delete, category deletion, and tag deletion are all implemented in the store and (for some) in the client's request layer, but no control in the shipped interface invokes them — while the sidebar still exposes an Archive destination that nothing can populate.
- **Activity log has no viewer.** The audit trail is fully written and readable but never surfaced in the interface.
- **Calendar scheduling uses a blocking browser prompt**, which cannot carry priority, category, or capture tokens.
- **Board quick-add into the Done column** creates a completed-status task without a completion timestamp, so it is invisible to streak and heatmap computation.

### Conflicting evidence

- **Screenshots are not what they claim.** All six "view tour" screenshots are byte-identical copies of the same list-view capture. The documentation labels them as the Kanban board, matrix, calendar, analytics dashboard, and Pomodoro timer. They are therefore evidence only for the list view, sidebar, header, and quick-capture bar; every other view was specified from source.
- **Stored artifact vs. current defaults.** The committed database contains development leftovers from a manual session — ad-hoc tasks ("test", "todo", "mytask"), residue from the end-to-end test run, and twenty activity entries — alongside the seeded content. It is a captured working state, not a reproducible fixture, and the seed routine will not recreate it.
- **Plan document vs. implementation.** The repository's implementation plan describes a different system: an in-memory JSON store (the project uses a durable relational store), a utility-CSS framework (the project uses hand-written CSS), and resource/stream paths that do not match the implemented ones. It also presents velocity and quadrant-scoring formulas that appear nowhere in the code.

### Documented but unimplemented intent

- The accompanying paper, abstract, and audit report describe a machine-learning system: leakage-free feature transformers, cross-validation folds, walk-forward validation, seed pinning across numerical/ML libraries, model cards, gradient-boosting and transformer baselines, and a Python service layer. None of this exists in the project, which contains no model, no dataset, and no Python. These documents also assert measured results — a 10,000-mutation benchmark, sub-10 ms interaction latency, a 38% reduction in task-creation friction, zero leakage, and a 99.3% compliance grade — with no benchmark harness, measurement code, or result artifact anywhere in the repository to support them. They should be read as generic boilerplate, not as a specification of this project.
- Documentation describes the capture syntax as supporting clock times (e.g. `@5pm`, "tomorrow @2pm" in the placeholder text). Only calendar-date resolution is implemented; the task model has no time-of-day component.
- The bundled agent skill documents (dashboard specification, visualization builder, TypeScript patterns), duplicated verbatim in two directories, are development-support material with no runtime role.

### Unresolved uncertainty

- Whether the quick-capture tag path and the detail-editor state declaration actually fault at runtime cannot be established without execution; both are recorded as apparent defects on static grounds.
- The provenance of the correctly-tagged task in the committed database is unexplained given the creation path's inspected behavior.

---

## 8. Acceptance Checklist

- [ ] `F01` — Tasks, subtasks, tags, and categories persist across restarts with the specified attributes; first run seeds demonstration content.
- [ ] `F02` — Free-text capture extracts priority, tags, due date, and estimate, previews them live, and uses the residual text as the title.
- [ ] `F03` — All five views are reachable and share the active filter, search, and sort.
- [ ] `F04` — List view groups tasks into the six collapsible date buckets with the specified row information.
- [ ] `F05` — Board presents four workflow columns with counts; dragging changes status; per-column quick-add works.
- [ ] `F06` — Matrix derives quadrant membership from priority and due date, and a drop places the task into the target quadrant.
- [ ] `F07` — Calendar shows due-date chips by day with month navigation and click-to-schedule.
- [ ] `F08` — Analytics reports completion rate, streak, productivity index, logged focus time, a 30-day heatmap, and priority/category breakdowns, recomputed from current data.
- [ ] `F09` — Detail editor edits every task property plus subtasks and tags in one save.
- [ ] `F10` — Completion toggling stamps/clears the completion time and produces audio-visual feedback.
- [ ] `F11` — A change in one client appears in other open clients without reload.
- [ ] `F12` — Smart lists, category and tag filters, search, and sorting compose correctly, with pinned tasks ordered first.
- [ ] `F13` — Task mutations produce retrievable, dated audit entries.
- [ ] `F14` — Pomodoro runs its three modes and attributes the focused time to the bound task.
- [ ] `S01` — Multi-select drives bulk complete, priority, category, and trash actions.
- [ ] `S02` — Command palette searches commands, tasks, and categories with keyboard navigation.
- [ ] `S03` — Global shortcuts work outside text fields and are documented in-app.
- [ ] `S04` / `S05` — Categories and tags can be created and reviewed with counts; tag names are normalized and duplicates reuse the existing tag.
- [ ] `S06` / `S07` — Theme, accent, and mute preferences apply immediately and persist per browser.
- [ ] `S08` — Pinning is settable from a row and the editor, and affects ordering.
- [ ] `S09` — Delete trashes an active task and purges one already trashed; archived and trashed tasks are excluded from views and telemetry.
- [ ] `S10` / `S11` — Sidebar collapses; loading and empty states appear where specified.
