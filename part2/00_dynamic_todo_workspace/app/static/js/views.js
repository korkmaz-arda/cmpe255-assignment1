// Renderers for the five workspace views. They only present server-derived
// data (buckets, quadrants, overdue flags, month grid, telemetry) and turn
// gestures into API calls.

import { api } from "./api.js";
import { categoryIcon, clear, h, icon, shortDate } from "./dom.js";
import { PRIORITY_LABELS, STATUS_LABELS, actions, categoryById, state, taskById } from "./state.js";

// ------------------------------------------------------------------ shared
function categoryChip(task) {
  const cat = categoryById(task.category_id);
  if (!cat) return null;
  return h("span", { class: "chip chip-category", style: { "--chip": cat.color } }, categoryIcon(cat.icon, 12), cat.name);
}

function dueChip(task) {
  if (!task.due_date) return null;
  const cls = task.overdue ? "chip chip-due overdue" : task.due_today ? "chip chip-due today" : "chip chip-due";
  return h("span", { class: cls, title: `Due ${task.due_date}` },
    task.overdue ? icon("alert", 12) : icon("calendar", 12),
    task.due_today ? "Today" : shortDate(task.due_date));
}

function timeChip(task) {
  if (task.time_spent_minutes > 0) {
    const est = task.estimated_minutes ? `${task.estimated_minutes}m` : "–";
    return h("span", { class: "chip chip-time", title: "Time spent / estimate" }, icon("clock", 12),
      `${task.time_spent_minutes}m / ${est}`);
  }
  if (task.estimated_minutes) {
    return h("span", { class: "chip chip-time", title: "Estimate" }, icon("clock", 12), `${task.estimated_minutes}m`);
  }
  return null;
}

function subtaskChip(task) {
  if (!task.subtasks_total) return null;
  const pct = Math.round((task.subtasks_done / task.subtasks_total) * 100);
  return h("span", { class: "chip chip-subtasks", title: "Subtasks done / total" },
    `${task.subtasks_done}/${task.subtasks_total}`,
    h("span", { class: "mini-bar" }, h("span", { style: { width: `${pct}%` } })));
}

const tagChips = (task) => task.tags.map((t) => h("span", { class: "chip chip-tag", style: { "--chip": t.color } }, `#${t.name}`));

function emptyState(title, hint) {
  return h("div", { class: "empty-state" }, icon("inbox", 36), h("h3", {}, title), h("p", {}, hint));
}

function rowActions(task) {
  const scope = state.nav.scope;
  const btn = (name, label, onclick, cls = "") =>
    h("button", { class: `icon-btn ${cls}`, title: label, "aria-label": label, onclick: (e) => { e.stopPropagation(); onclick(); } }, icon(name, 15));
  if (scope === "trash") {
    return [btn("restore", "Restore", () => actions.restore(task)),
      btn("trash", "Delete forever", () => actions.deleteTask(task), "danger")];
  }
  return [
    btn("pin", task.pinned ? "Unpin" : "Pin to top", () => actions.togglePin(task), task.pinned ? "active" : ""),
    btn("edit", "Open details", () => actions.openEditor(task.id)),
    scope === "archive" ? btn("restore", "Unarchive", () => actions.unarchive(task))
      : btn("archive", "Archive", () => actions.archive(task)),
    btn("trash", "Move to trash", () => actions.deleteTask(task), "danger"),
  ];
}

function completeButton(task) {
  const done = task.status === "completed";
  return h("button", {
    class: `complete-check${done ? " done" : ""}`, role: "checkbox", "aria-checked": String(done),
    "aria-label": done ? "Mark as not done" : "Mark as done", disabled: state.nav.scope === "trash",
    onclick: (e) => { e.stopPropagation(); actions.toggleComplete(task); },
  }, done ? icon("check", 13) : null);
}

// -------------------------------------------------------------------- list
const BUCKETS = [
  ["overdue", "Overdue"], ["today", "Due Today"], ["tomorrow", "Due Tomorrow"],
  ["upcoming", "Upcoming & Scheduled"], ["no_date", "No Due Date"], ["completed", "Completed"],
];

function taskRow(task) {
  const selected = state.selected.has(task.id);
  return h("div", {
    class: `task-row priority-${task.priority}${task.status === "completed" ? " is-done" : ""}${selected ? " selected" : ""}`,
    "data-task-id": task.id, onclick: () => actions.openEditor(task.id),
  },
  h("input", { type: "checkbox", class: "select-box", checked: selected, "aria-label": "Select task",
    onclick: (e) => e.stopPropagation(), onchange: () => actions.toggleSelect(task.id) }),
  completeButton(task),
  h("div", { class: "row-main" },
    h("div", { class: "row-title" }, h("span", { class: "title-text" }, task.title),
      task.pinned ? h("span", { class: "pin-marker", title: "Pinned" }, icon("pin", 12)) : null),
    h("div", { class: "row-meta" },
      h("span", { class: `chip chip-priority p-${task.priority}` }, PRIORITY_LABELS[task.priority]),
      dueChip(task), timeChip(task), categoryChip(task), subtaskChip(task), tagChips(task))),
  h("div", { class: "row-actions" }, rowActions(task)));
}

export function renderList(root) {
  const tasks = state.server.tasks;
  if (!tasks.length) {
    const hint = state.nav.scope === "trash" ? "Trash is empty."
      : state.nav.scope === "archive" ? "Archived tasks appear here."
        : state.search ? "No tasks match your search." : "Capture a task above, e.g. “Plan sprint !high #work @friday”.";
    return clear(root, emptyState("No tasks here", hint));
  }
  const allSelected = tasks.every((t) => state.selected.has(t.id));
  const toolbar = h("div", { class: "list-toolbar" },
    h("label", { class: "select-all" },
      h("input", { type: "checkbox", checked: allSelected, onchange: () => actions.selectAll(!allSelected) }),
      allSelected ? "Deselect all" : "Select all"),
    h("span", { class: "muted" }, `${tasks.length} task${tasks.length === 1 ? "" : "s"}`));
  const groups = BUCKETS.map(([key, label]) => {
    const items = tasks.filter((t) => t.bucket === key);
    if (!items.length) return null;
    const collapsed = state.collapsedBuckets.has(key);
    return h("section", { class: `bucket bucket-${key}`, "data-bucket": key },
      h("button", { class: "bucket-header", "aria-expanded": String(!collapsed),
        onclick: () => { collapsed ? state.collapsedBuckets.delete(key) : state.collapsedBuckets.add(key); actions.render(); } },
      h("span", { class: `chevron${collapsed ? " collapsed" : ""}` }, icon("chevronDown", 14)),
      h("span", { class: "bucket-dot" }), label, h("span", { class: "count" }, items.length)),
      collapsed ? null : h("div", { class: "bucket-body" }, items.map(taskRow)));
  });
  clear(root, toolbar, groups);
}

// ------------------------------------------------------------------- board
const COLUMNS = ["todo", "in_progress", "review", "completed"];
let dragId = null;

function card(task, extra = {}) {
  const cat = categoryById(task.category_id);
  return h("article", {
    class: `card priority-${task.priority}${task.status === "completed" ? " is-done" : ""}`,
    draggable: state.nav.scope === "active", "data-task-id": task.id,
    ondragstart: (e) => {
      dragId = task.id;
      state.dragging = true; // app.js defers re-renders until the drag ends
      e.dataTransfer.setData("text/plain", String(task.id));
      e.dataTransfer.effectAllowed = "move";
      e.currentTarget.classList.add("dragging");
    },
    ondragend: (e) => {
      e.currentTarget.classList.remove("dragging");
      dragId = null;
      state.dragging = false;
      actions.dragEnded();
    },
    onclick: () => actions.openEditor(task.id),
    ...extra,
  },
  h("div", { class: "card-top" },
    h("span", { class: `chip chip-priority p-${task.priority}` }, PRIORITY_LABELS[task.priority]),
    cat ? h("span", { class: "card-cat", style: { color: cat.color } }, categoryIcon(cat.icon, 12), cat.name) : null,
    task.pinned ? h("span", { class: "pin-marker" }, icon("pin", 12)) : null),
  h("div", { class: "card-title" }, task.title),
  task.description ? h("div", { class: "card-desc" }, task.description.length > 90 ? `${task.description.slice(0, 90)}…` : task.description) : null,
  h("div", { class: "row-meta" }, dueChip(task), subtaskChip(task), tagChips(task)));
}

function draggedId(e) {
  const raw = e.dataTransfer.getData("text/plain");
  return raw ? Number(raw) : dragId;
}

function insertionIndex(container, clientY) {
  const cards = [...container.querySelectorAll(".card:not(.dragging)")];
  const i = cards.findIndex((c) => {
    const box = c.getBoundingClientRect();
    return clientY < box.top + box.height / 2;
  });
  return i === -1 ? cards.length : i;
}

async function dropOnColumn(status, container, e) {
  const id = draggedId(e);
  const task = taskById(id);
  if (!task) return;
  const column = state.server.tasks.filter((t) => t.status === status && t.id !== id);
  const index = insertionIndex(container, e.clientY);
  const prev = column[index - 1];
  const next = column[index];
  let items;
  if (!prev && !next) items = [{ id, order_index: task.order_index }];
  else if (!prev) items = [{ id, order_index: next.order_index - 1000 }];
  else if (!next) items = [{ id, order_index: prev.order_index + 1000 }];
  else if (next.order_index - prev.order_index > 1) items = [{ id, order_index: Math.floor((prev.order_index + next.order_index) / 2) }];
  else {
    const ordered = [...column.slice(0, index), task, ...column.slice(index)];
    const base = Math.min(...ordered.map((t) => t.order_index));
    items = ordered.map((t, i) => ({ id: t.id, order_index: base + i * 1000 }));
  }
  const becameDone = status === "completed" && task.status !== "completed";
  try {
    await api.reorder(items, status);
    if (becameDone) actions.completionFeedback();
  } catch (err) {
    actions.toast(err.message, "error");
  }
  actions.refresh();
}

export function renderBoard(root) {
  const columns = COLUMNS.map((status) => {
    const items = state.server.tasks.filter((t) => t.status === status);
    const body = h("div", { class: "column-body" }, items.length ? items.map((t) => card(t)) : h("div", { class: "column-empty" }, "Drop tasks here"));
    const input = h("input", { class: "quick-add", placeholder: "+ Add task", "aria-label": `Add task to ${STATUS_LABELS[status]}`,
      disabled: state.nav.scope !== "active",
      onkeydown: async (e) => {
        if (e.key !== "Enter" || !e.target.value.trim()) return;
        const text = e.target.value;
        try {
          await actions.capture({ capture: text, status });
          e.target.value = "";
        } catch (err) { actions.toast(err.message, "error"); }
      } });
    return h("section", {
      class: `column column-${status}`, "data-status": status,
      ondragover: (e) => { e.preventDefault(); e.currentTarget.classList.add("drag-over"); },
      ondragleave: (e) => { if (!e.currentTarget.contains(e.relatedTarget)) e.currentTarget.classList.remove("drag-over"); },
      ondrop: (e) => { e.preventDefault(); e.currentTarget.classList.remove("drag-over"); dropOnColumn(status, body, e); },
    },
    h("header", { class: "column-header" }, h("span", { class: "column-dot" }), STATUS_LABELS[status], h("span", { class: "count" }, items.length)),
    input, body);
  });
  clear(root, h("div", { class: "board" }, columns));
}

// ------------------------------------------------------------------ matrix
const QUADRANTS = [
  ["do_first", "Do First", "Urgent and important: crises and deadlines", "Nothing on fire. Nice."],
  ["schedule", "Schedule", "Important, not urgent: plan time for it", "Drop strategic work here"],
  ["delegate", "Delegate", "Urgent, not important: interruptions", "No interruptions queued"],
  ["eliminate", "Eliminate / Backlog", "Neither urgent nor important", "Backlog is empty"],
];

export function renderMatrix(root) {
  const quads = QUADRANTS.map(([key, title, subtitle, emptyHint]) => {
    const items = state.server.tasks.filter((t) => t.quadrant === key);
    return h("section", {
      class: `quadrant quadrant-${key}`, "data-quadrant": key,
      ondragover: (e) => { e.preventDefault(); e.currentTarget.classList.add("drag-over"); },
      ondragleave: (e) => { if (!e.currentTarget.contains(e.relatedTarget)) e.currentTarget.classList.remove("drag-over"); },
      ondrop: async (e) => {
        e.preventDefault();
        e.currentTarget.classList.remove("drag-over");
        const id = draggedId(e);
        const task = taskById(id);
        if (!task || task.quadrant === key) return;
        try { await api.matrixDrop(id, key); } catch (err) { actions.toast(err.message, "error"); }
        actions.refresh();
      },
    },
    h("header", {}, h("div", {}, h("h3", {}, title), h("p", { class: "muted" }, subtitle)), h("span", { class: "count" }, items.length)),
    h("div", { class: "quadrant-body" }, items.length ? items.map((t) => card(t)) : h("div", { class: "column-empty" }, emptyHint)));
  });
  clear(root,
    h("p", { class: "view-note muted" }, "Quadrants are derived from each task: urgent = priority Urgent or due today/overdue; important = priority Urgent or High. Dragging a task changes exactly those attributes. Completed tasks are not shown."),
    h("div", { class: "matrix" }, quads));
}

// ---------------------------------------------------------------- calendar
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

// The open composer survives live-sync re-renders of the calendar.
let composer = null; // { date, value }

export function closeDayComposer() {
  composer = null;
  document.querySelectorAll(".day-composer").forEach((el) => el.remove());
}

function openDayComposer(cell, iso, value = "") {
  if (cell.querySelector(".day-composer")) return;
  closeDayComposer();
  composer = { date: iso, value };
  const input = h("input", { class: "day-composer-input", placeholder: "Task title (tokens allowed)", "aria-label": `New task on ${iso}`,
    value, oninput: () => { if (composer) composer.value = input.value; } });
  const form = h("form", {
    class: "day-composer", onclick: (e) => e.stopPropagation(),
    onsubmit: async (e) => {
      e.preventDefault();
      if (!input.value.trim()) return;
      try {
        const text = input.value;
        closeDayComposer();
        await actions.capture({ capture: text, default_due: iso });
      } catch (err) { actions.toast(err.message, "error"); }
    },
  }, input, h("div", { class: "hint" }, "Enter to add · Esc to cancel"));
  input.addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); closeDayComposer(); } });
  cell.append(form);
  input.focus();
}

export function renderCalendar(root) {
  const grid = state.server.calendar;
  if (!grid) return clear(root, h("div", { class: "loading" }, "Loading calendar…"));
  const byDay = new Map();
  for (const t of state.server.tasks) {
    if (!t.due_date) continue;
    if (!byDay.has(t.due_date)) byDay.set(t.due_date, []);
    byDay.get(t.due_date).push(t);
  }
  const nav = (target) => { state.calendarMonth = target; actions.refresh(); };
  const header = h("div", { class: "calendar-header" },
    h("button", { class: "icon-btn", "aria-label": "Previous month", title: "Previous month", onclick: () => nav(grid.prev) }, icon("chevronLeft")),
    h("h2", { class: "calendar-title" }, grid.label),
    h("button", { class: "icon-btn", "aria-label": "Next month", title: "Next month", onclick: () => nav(grid.next) }, icon("chevronRight")),
    h("button", { class: "btn", onclick: () => nav(null) }, "Today"));
  const cells = grid.weeks.flat().map((day) => {
    const tasks = byDay.get(day.date) || [];
    const cell = h("div", {
      class: `day${day.in_month ? "" : " outside"}${day.is_today ? " is-today" : ""}`, "data-date": day.date,
      title: state.nav.scope === "active" ? "Click empty space to add a task on this day" : "",
      onclick: (e) => {
        if (state.nav.scope !== "active") return;
        if (e.target === e.currentTarget || e.target.closest(".day-head")) openDayComposer(e.currentTarget, day.date);
      },
    },
    h("div", { class: "day-head" }, h("span", { class: "day-num" }, day.day), tasks.length ? h("span", { class: "count" }, tasks.length) : null),
    tasks.map((t) => h("button", {
      class: `cal-chip p-${t.priority}${t.status === "completed" ? " is-done" : ""}`, "data-task-id": t.id, title: t.title,
      onclick: (e) => { e.stopPropagation(); actions.openEditor(t.id); },
    }, t.title)));
    return cell;
  });
  clear(root, header, h("div", { class: "calendar-grid" }, WEEKDAYS.map((d) => h("div", { class: "weekday" }, d)), cells));
  if (composer) {
    const cell = root.querySelector(`.day[data-date="${composer.date}"]`);
    const { date, value } = composer;
    composer = null;
    if (cell) openDayComposer(cell, date, value);
  }
}

// --------------------------------------------------------------- analytics
function metricTile(label, value, sub, help) {
  return h("div", { class: "metric", title: help },
    h("div", { class: "metric-label" }, label, h("span", { class: "help", "aria-label": help }, "?")),
    h("div", { class: "metric-value" }, value), h("div", { class: "metric-sub muted" }, sub));
}

export function renderAnalytics(root) {
  const a = state.server.analytics;
  if (!a) return clear(root, h("div", { class: "loading" }, "Loading telemetry…"));
  const empty = a.total === 0; // no active tasks: ratio and index are undefined, not 0 / 20
  const tiles = h("div", { class: "metrics" },
    metricTile("Completion velocity", empty ? "—" : `${a.completion_rate}%`, empty ? "no active tasks yet" : `${a.completed} of ${a.total} tasks completed`,
      "Completed tasks ÷ all active (not archived, not trashed) tasks, rounded."),
    metricTile("Active streak", `${a.streak} day${a.streak === 1 ? "" : "s"}`, "consecutive days with a completion",
      "Counted back from today within the last 30 days. No completion yet today does not break it; any earlier empty day does. Based on current completion times, so reopening a task removes its day."),
    metricTile("Productivity index", empty ? "—" : `${a.productivity_index}/100`,
      empty ? "needs at least one active task" : "heuristic from velocity, streak and overdue tasks",
      "Rough composite, not a validated measure: round(velocity × 0.7) + min(streak × 5, 20) − min(overdue × 10, 30) + 20, kept within 10–100."),
    metricTile("Focus time logged", `${a.focus_hours.toFixed(1)} h`, `${a.focus_minutes} minutes total`,
      "Sum of time-spent minutes on active tasks, including Pomodoro focus sessions."));

  const legendCell = (lvl) => h("span", { class: `heat heat-${lvl}` });
  const heatmap = h("section", { class: "panel" },
    h("h3", {}, "30-day activity"),
    h("div", { class: "heatmap" }, a.heatmap.map((d) => h("div", {
      class: `heat heat-${d.level}`, "data-date": d.date,
      title: `${d.date}: ${d.completed} completed, ${d.created} created`,
      "aria-label": `${d.date}: ${d.completed} completed, ${d.created} created`,
    }))),
    h("div", { class: "legend muted" }, "Less", [0, 1, 2, 3, 4].map(legendCell), "More",
      h("span", { class: "legend-note" }, "Shade: 4+ / 2+ / 1 completions; faintest = tasks created only")));

  const priorities = h("section", { class: "panel" },
    h("h3", {}, "Completion by priority"),
    a.priorities.map((p) => h("div", { class: "bar-row", "data-priority": p.priority },
      h("span", { class: `chip chip-priority p-${p.priority}` }, PRIORITY_LABELS[p.priority]),
      h("div", { class: "bar", title: `${p.completion_pct}% of ${p.priority} tasks completed` }, h("span", { class: `fill p-${p.priority}`, style: { width: `${p.completion_pct}%` } })),
      h("span", { class: "bar-label" }, `${p.completed}/${p.total} · ${p.completion_pct}%`))),
    h("p", { class: "muted small" }, "Bar length = share of that priority's tasks that are completed."));

  const maxTotal = Math.max(1, ...a.categories.map((c) => c.total));
  const categories = h("section", { class: "panel" },
    h("h3", {}, "Tasks by category"),
    a.categories.length ? a.categories.map((c) => h("div", { class: "bar-row" },
      h("span", { class: "cat-name", style: { color: c.color } }, c.name),
      h("div", { class: "bar stacked", title: `${c.completed} completed of ${c.total}` },
        h("span", { class: "fill total", style: { width: `${(c.total / maxTotal) * 100}%`, background: c.color } }),
        h("span", { class: "fill done", style: { width: `${(c.completed / maxTotal) * 100}%`, background: c.color } })),
      h("span", { class: "bar-label" }, `${c.completed}/${c.total}`))) : h("p", { class: "muted" }, "No categories yet."),
    h("p", { class: "muted small" }, "Faint bar = total tasks (relative to largest category); solid part = completed."));

  const statuses = h("section", { class: "panel" }, h("h3", {}, "Workflow status"),
    h("div", { class: "status-counts" }, Object.entries(a.status_counts).map(([s, n]) =>
      h("div", { class: "status-count" }, h("span", { class: "metric-value small" }, n), h("span", { class: "muted" }, STATUS_LABELS[s]))),
    h("div", { class: "status-count" }, h("span", { class: "metric-value small danger" }, a.overdue), h("span", { class: "muted" }, "Overdue"))));

  clear(root,
    h("p", { class: "view-note muted" }, `Computed from the current task store as of ${a.today} (the server's date). Archived and trashed tasks are excluded. Independent of the sidebar filter.`),
    tiles, heatmap, h("div", { class: "panel-grid" }, priorities, categories), statuses);
}
