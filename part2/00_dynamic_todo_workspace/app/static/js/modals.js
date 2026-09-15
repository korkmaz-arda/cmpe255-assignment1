// Overlays: task detail editor, tag manager, activity log, shortcuts sheet,
// command palette, and the Pomodoro focus timer.

import { api } from "./api.js";
import { categoryIcon, clear, h, icon } from "./dom.js";
import { confetti, play } from "./feedback.js";
import { PRIORITY_LABELS, STATUS_LABELS, actions, state } from "./state.js";

// ------------------------------------------------------------ modal stack
const stack = [];

export function openModal(content, { onClose, className = "", label = "Dialog" } = {}) {
  const previousFocus = document.activeElement;
  const entry = {};
  const overlay = h("div", { class: `overlay ${className}`, onmousedown: (e) => { if (e.target === overlay) entry.close(); } },
    h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": label }, content));
  entry.overlay = overlay;
  entry.close = () => {
    const i = stack.indexOf(entry);
    if (i === -1) return;
    stack.splice(i, 1);
    overlay.remove();
    if (onClose) onClose();
    if (previousFocus && previousFocus.focus) previousFocus.focus();
  };
  stack.push(entry);
  document.getElementById("overlay-root").append(overlay);
  const focusable = overlay.querySelector("[autofocus], input, textarea, select, button");
  if (focusable) focusable.focus();
  return entry.close;
}

export function closeTopOverlay() {
  const top = stack[stack.length - 1];
  if (!top) return false;
  top.close();
  return true;
}

export const hasOverlay = () => stack.length > 0;
const isOpen = (cls) => stack.some((e) => e.overlay.classList.contains(cls));
const closeByClass = (cls) => stack.filter((e) => e.overlay.classList.contains(cls)).forEach((e) => e.close());

function modalHeader(title, close) {
  return h("header", { class: "modal-header" }, h("h2", {}, title),
    h("button", { class: "icon-btn", "aria-label": "Close", title: "Close (Esc)", onclick: () => close() }, icon("x")));
}

const option = (value, label, current) => h("option", { value, selected: String(current ?? "") === String(value) }, label);

// ------------------------------------------------------------ task editor
export async function openEditor(taskId) {
  let task;
  try {
    task = await api.getTask(taskId); // always edit the server's current values
  } catch (err) {
    actions.toast(err.message, "error");
    return;
  }
  const draft = {
    subtasks: task.subtasks.map((s) => ({ title: s.title, completed: s.completed })),
    tagIds: new Set(task.tags.map((t) => t.id)),
    pinned: task.pinned,
  };
  let close;
  const title = h("input", { class: "input title-input", value: task.title, "aria-label": "Title", autofocus: true });
  const description = h("textarea", { class: "input", rows: 3, placeholder: "Description", "aria-label": "Description" });
  description.value = task.description;
  const status = h("select", { class: "input", "aria-label": "Status" }, Object.entries(STATUS_LABELS).map(([v, l]) => option(v, l, task.status)));
  const priority = h("select", { class: "input", "aria-label": "Priority" }, Object.entries(PRIORITY_LABELS).map(([v, l]) => option(v, l, task.priority)));
  const category = h("select", { class: "input", "aria-label": "Category" }, option("", "No category", task.category_id),
    state.server.categories.map((c) => option(c.id, c.name, task.category_id)));
  const due = h("input", { class: "input", type: "date", value: task.due_date || "", "aria-label": "Due date" });
  const estimate = h("input", { class: "input", type: "number", min: 0, step: 1, value: task.estimated_minutes ?? "", "aria-label": "Estimated minutes" });
  const spent = h("input", { class: "input", type: "number", min: 0, step: 1, value: task.time_spent_minutes, "aria-label": "Time spent minutes" });
  const pinBtn = h("button", { type: "button", class: "btn" });
  const renderPin = () => { clear(pinBtn, icon("pin", 14), draft.pinned ? "Pinned" : "Pin to top"); pinBtn.classList.toggle("active", draft.pinned); };
  pinBtn.addEventListener("click", () => { draft.pinned = !draft.pinned; renderPin(); });
  renderPin();

  const subList = h("div", { class: "subtask-list" });
  const counter = h("span", { class: "muted" });
  const renderSubtasks = () => {
    const done = draft.subtasks.filter((s) => s.completed).length;
    counter.textContent = `${done}/${draft.subtasks.length} done`;
    clear(subList, draft.subtasks.map((s, i) => h("div", { class: "subtask" },
      h("input", { type: "checkbox", checked: s.completed, "aria-label": `Toggle ${s.title}`, onchange: () => { s.completed = !s.completed; renderSubtasks(); } }),
      h("span", { class: s.completed ? "done" : "" }, s.title),
      h("button", { type: "button", class: "icon-btn", "aria-label": `Remove ${s.title}`, onclick: () => { draft.subtasks.splice(i, 1); renderSubtasks(); } }, icon("x", 13)))));
  };
  const newSub = h("input", { class: "input", placeholder: "Add a subtask and press Enter", "aria-label": "New subtask",
    onkeydown: (e) => {
      if (e.key !== "Enter") return;
      e.preventDefault();
      if (!newSub.value.trim()) return;
      draft.subtasks.push({ title: newSub.value.trim(), completed: false });
      newSub.value = "";
      renderSubtasks();
    } });
  renderSubtasks();

  const tagBox = h("div", { class: "tag-picker" });
  const renderTags = () => clear(tagBox, state.server.tags.length ? state.server.tags.map((t) => h("button", {
    type: "button", class: `chip chip-tag toggle${draft.tagIds.has(t.id) ? " on" : ""}`, style: { "--chip": t.color },
    "aria-pressed": String(draft.tagIds.has(t.id)),
    onclick: () => { draft.tagIds.has(t.id) ? draft.tagIds.delete(t.id) : draft.tagIds.add(t.id); renderTags(); },
  }, `#${t.name}`)) : h("span", { class: "muted" }, "No tags yet. Create them from the sidebar."));
  renderTags();

  const error = h("div", { class: "form-error", role: "alert" });
  const num = (input) => (input.value === "" ? null : Number(input.value));

  async function save() {
    const payload = {
      title: title.value, description: description.value, status: status.value, priority: priority.value,
      category_id: category.value ? Number(category.value) : null, due_date: due.value || null,
      estimated_minutes: num(estimate), time_spent_minutes: num(spent) ?? 0, pinned: draft.pinned,
      subtasks: draft.subtasks, tag_ids: [...draft.tagIds],
    };
    if (!payload.title.trim()) { error.textContent = "Title is required."; return; }
    try {
      const saved = await api.updateTask(task.id, payload);
      if (saved.status === "completed" && task.status !== "completed") actions.completionFeedback();
      close();
      actions.refresh();
    } catch (err) {
      error.textContent = err.message;
    }
  }

  const field = (label, control) => h("label", { class: "field" }, h("span", {}, label), control);
  const content = h("form", { class: "editor", onsubmit: (e) => { e.preventDefault(); save(); } },
    modalHeader("Task details", () => close()),
    h("div", { class: "modal-body" },
      h("div", { class: "muted small" }, `Task #${task.id}${task.completed_at ? ` · completed ${task.completed_at.replace("T", " ")}` : ""}`),
      title, description,
      h("div", { class: "field-grid" }, field("Status", status), field("Priority", priority), field("Category", category),
        field("Due date", due), field("Estimate (min)", estimate), field("Time spent (min)", spent)),
      h("div", { class: "section-title" }, "Subtasks ", counter), subList, newSub,
      h("div", { class: "section-title" }, "Tags"), tagBox, error),
    h("footer", { class: "modal-footer" },
      pinBtn,
      h("button", { type: "button", class: "btn danger", onclick: async () => { if (await actions.deleteTask(task)) close(); } }, icon("trash", 14), task.deleted ? "Delete forever" : "Move to trash"),
      h("span", { class: "spacer" }),
      h("button", { type: "button", class: "btn", onclick: () => close() }, "Cancel"),
      h("button", { type: "submit", class: "btn primary" }, "Save")));
  close = openModal(content, { className: "editor-overlay", label: "Task details" });
}

// ------------------------------------------------------------ tag manager
export function openTagManager() {
  const palette = state.server.palettes.tag;
  let color = palette[0];
  let close;
  const name = h("input", { class: "input", placeholder: "Tag name", "aria-label": "Tag name", autofocus: true });
  const swatches = h("div", { class: "swatches" });
  const renderSwatches = () => clear(swatches, palette.map((c) => h("button", {
    type: "button", class: `swatch${c === color ? " on" : ""}`, style: { background: c }, "aria-label": `Color ${c}`,
    onclick: () => { color = c; renderSwatches(); },
  })));
  renderSwatches();
  const list = h("div", { class: "tag-list" });
  const message = h("div", { class: "muted small", role: "status" });
  const renderList = () => clear(list, state.server.tags.length ? state.server.tags.map((t) => h("div", { class: "tag-item" },
    h("span", { class: "chip chip-tag", style: { "--chip": t.color } }, `#${t.name}`),
    h("span", { class: "muted" }, `used by ${t.usage} task${t.usage === 1 ? "" : "s"}`),
    h("button", { class: "icon-btn danger", "aria-label": `Delete tag ${t.name}`, title: "Delete tag (tasks are kept)",
      onclick: async () => {
        if (!confirm(`Delete tag #${t.name}? It will be removed from its tasks; the tasks are kept.`)) return;
        try { await api.deleteTag(t.id); await actions.refresh(); renderList(); } catch (err) { actions.toast(err.message, "error"); }
      } }, icon("trash", 14)))) : h("p", { class: "muted" }, "No tags yet."));
  renderList();
  const content = h("div", {},
    modalHeader("Tags", () => close()),
    h("div", { class: "modal-body" },
      h("form", { class: "tag-form", onsubmit: async (e) => {
        e.preventDefault();
        if (!name.value.trim()) return;
        try {
          const tag = await api.createTag(name.value, color);
          message.textContent = tag.created ? `Created #${tag.name}.` : `#${tag.name} already exists; kept the existing tag.`;
          name.value = "";
          await actions.refresh();
          renderList();
        } catch (err) { message.textContent = err.message; }
      } }, name, swatches, h("button", { class: "btn primary", type: "submit" }, "Create tag")),
      message, list));
  close = openModal(content, { className: "tags-overlay", label: "Tags" });
}

// ------------------------------------------------------------ activity log
export function openActivity() {
  let close;
  let limit = 50;
  const list = h("div", { class: "activity-list" }, h("div", { class: "loading" }, "Loading…"));
  const load = async () => {
    try {
      const entries = await api.activity(limit);
      clear(list, entries.length ? entries.map((e) => h("div", { class: "activity-item" },
        h("span", { class: `badge action-${e.action.replace(/\s+/g, "-")}` }, e.action),
        h("div", {}, h("div", {}, e.task_title, h("span", { class: "muted small" }, e.task_id ? ` #${e.task_id}` : "")),
          h("div", { class: "muted small" }, `${e.created_at.replace("T", " ")} · ${e.details}`)))) : h("p", { class: "muted" }, "No activity recorded."));
    } catch (err) { clear(list, h("p", { class: "form-error" }, err.message)); }
  };
  const limitSelect = h("select", { class: "input compact", "aria-label": "Entries to show", onchange: (e) => { limit = Number(e.target.value); load(); } },
    [20, 50, 100, 200].map((n) => option(n, `Last ${n}`, limit)));
  const content = h("div", {},
    modalHeader("Activity log", () => close()),
    h("div", { class: "modal-body" },
      h("div", { class: "row-between" }, limitSelect,
        h("button", { class: "btn danger", onclick: async () => {
          if (!confirm("Clear the entire activity log? This cannot be undone.")) return;
          try { await api.clearActivity(); load(); } catch (err) { actions.toast(err.message, "error"); }
        } }, "Clear log")),
      list));
  close = openModal(content, { className: "activity-overlay", label: "Activity log" });
  load();
}

// ------------------------------------------------------------ shortcuts
export function toggleShortcuts() {
  if (isOpen("shortcuts-overlay")) return closeByClass("shortcuts-overlay");
  let close;
  const row = (keys, text) => h("div", { class: "shortcut" }, h("span", {}, keys.map((k) => h("kbd", {}, k))), h("span", {}, text));
  const content = h("div", {},
    modalHeader("Keyboard shortcuts", () => close()),
    h("div", { class: "modal-body" },
      row(["Ctrl/⌘", "K"], "Toggle command palette"), row(["/"], "Focus search"), row(["?"], "Toggle this reference"),
      row(["Esc"], "Close any open overlay"),
      h("div", { class: "section-title" }, "Quick capture tokens"),
      row(["!urgent"], "Priority: !urgent !crit !u · !high !h !important · !medium !med !m · !low !l"),
      row(["#tag"], "Tag (any number; created if new)"),
      row(["~30m"], "Estimate: ~45, ~30m, ~1.5h (hours × 60)"),
      row(["@tomorrow"], "Due: @today @tom @yesterday @nextweek @2026-10-01 @fri (next Friday)"),
      h("p", { class: "muted small" }, "Recognized tokens are removed from the title. Only calendar dates are supported (no clock times).")));
  close = openModal(content, { className: "shortcuts-overlay", label: "Keyboard shortcuts" });
}

// ------------------------------------------------------------ command palette
export function togglePalette() {
  if (isOpen("palette-overlay")) return closeByClass("palette-overlay");
  let close;
  let highlighted = 0;
  let entries = [];
  const input = h("input", { class: "palette-input", placeholder: "Type a command, task, or category…", "aria-label": "Command palette", autofocus: true });
  const list = h("div", { class: "palette-list", role: "listbox" });

  const commands = () => [
    ...["list", "board", "matrix", "calendar", "analytics"].map((v) => ({ label: `Go to ${v[0].toUpperCase()}${v.slice(1)} view`, group: "View", icon: { list: "list", board: "board", matrix: "matrix", calendar: "calendar", analytics: "chart" }[v], run: () => actions.setView(v) })),
    { label: "Toggle dark / light theme", group: "Theme", icon: "moon", run: () => actions.toggleTheme() },
    ...["indigo", "cyan", "emerald", "amber", "rose"].map((a) => ({ label: `Accent: ${a}`, group: "Theme", icon: "palette", run: () => actions.setAccent(a) })),
    { label: "Open focus timer", group: "Tools", icon: "timer", run: () => openPomodoro() },
    { label: "Manage tags", group: "Tools", icon: "tag", run: () => openTagManager() },
    { label: "Show activity log", group: "Tools", icon: "activity", run: () => openActivity() },
    { label: "Keyboard shortcuts", group: "Tools", icon: "keyboard", run: () => toggleShortcuts() },
  ];

  const render = () => {
    const q = input.value.trim().toLowerCase();
    const cmds = commands().filter((c) => !q || c.label.toLowerCase().includes(q));
    const tasks = q ? state.server.tasks.filter((t) => t.title.toLowerCase().includes(q)).slice(0, 5)
      .map((t) => ({ label: t.title, group: "Task", icon: "edit", run: () => actions.openEditor(t.id) })) : [];
    const cats = q ? state.server.categories.filter((c) => c.name.toLowerCase().includes(q))
      .map((c) => ({ label: c.name, group: "Category", iconNode: categoryIcon(c.icon, 15), run: () => actions.setNav({ scope: "active", smart: "all", category: c.id }) })) : [];
    entries = [...cmds, ...tasks, ...cats];
    highlighted = Math.min(highlighted, Math.max(0, entries.length - 1));
    clear(list, entries.length ? entries.map((e, i) => h("div", {
      class: `palette-item${i === highlighted ? " highlighted" : ""}`, role: "option", "aria-selected": String(i === highlighted),
      onmousemove: () => { if (highlighted !== i) { highlighted = i; render(); } },
      onclick: () => runEntry(i),
    }, e.iconNode || icon(e.icon, 15), h("span", { class: "label" }, e.label), h("span", { class: "group muted" }, e.group)))
      : h("div", { class: "palette-empty muted" }, "No results"));
    const active = list.querySelector(".highlighted");
    if (active && active.scrollIntoView) active.scrollIntoView({ block: "nearest" });
  };
  const runEntry = (i) => {
    const entry = entries[i];
    if (!entry) return;
    close();
    entry.run();
  };
  input.addEventListener("input", () => { highlighted = 0; render(); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); highlighted = entries.length ? (highlighted + 1) % entries.length : 0; render(); }
    else if (e.key === "ArrowUp") { e.preventDefault(); highlighted = entries.length ? (highlighted - 1 + entries.length) % entries.length : 0; render(); }
    else if (e.key === "Enter") { e.preventDefault(); runEntry(highlighted); }
  });
  render();
  close = openModal(h("div", { class: "palette" }, h("div", { class: "palette-search" }, icon("search"), input), list),
    { className: "palette-overlay", label: "Command palette" });
}

// ------------------------------------------------------------ Pomodoro
// Countdown is measured from real elapsed wall-clock time (Date.now), so it
// stays correct in background tabs. The logged time is the time actually
// focused: a full session logs its measured duration, and resetting,
// switching mode, re-binding, or closing mid-session logs the partial time
// (rounded to whole minutes; under 1 minute logs nothing).
const MODES = {
  focus: { label: "Focus", minutes: 25 },
  short: { label: "Short Break", minutes: 5 },
  long: { label: "Long Break", minutes: 15 },
};
const pomodoro = { sessions: 0, boundTaskId: null };

export function openPomodoro() {
  if (isOpen("pomodoro-overlay")) return;
  let close;
  const t = { mode: "focus", elapsedBefore: 0, runningSince: null, creditedMs: 0, interval: null };
  const duration = () => MODES[t.mode].minutes * 60000;
  const elapsed = () => t.elapsedBefore + (t.runningSince === null ? 0 : Date.now() - t.runningSince);

  const display = h("div", { class: "timer-display", "aria-live": "polite" });
  const bar = h("span", {});
  const startBtn = h("button", { class: "btn primary big", onclick: () => (t.runningSince === null ? start() : pause()) });
  const message = h("div", { class: "timer-message", role: "status" });
  const sessionCount = h("span", { class: "muted" });
  const modeButtons = h("div", { class: "segmented" });
  const taskSelect = h("select", { class: "input", "aria-label": "Bind to task",
    onchange: async () => {
      await creditPartial();
      pomodoro.boundTaskId = taskSelect.value ? Number(taskSelect.value) : null;
    } }, option("", "No task bound", ""));

  function paint() {
    const remaining = Math.max(0, duration() - elapsed());
    const secs = Math.ceil(remaining / 1000);
    display.textContent = `${String(Math.floor(secs / 60)).padStart(2, "0")}:${String(secs % 60).padStart(2, "0")}`;
    bar.style.width = `${Math.min(100, (elapsed() / duration()) * 100)}%`;
    startBtn.textContent = t.runningSince === null ? (elapsed() > 0 ? "Resume" : "Start") : "Pause";
    sessionCount.textContent = `Focus sessions completed: ${pomodoro.sessions}`;
    clear(modeButtons, Object.entries(MODES).map(([key, m]) => h("button", {
      class: `seg${key === t.mode ? " on" : ""}`, "aria-pressed": String(key === t.mode), onclick: () => switchMode(key),
    }, `${m.label} ${m.minutes}m`)));
  }

  async function credit(ms) {
    if (t.mode !== "focus" || !pomodoro.boundTaskId || ms < 60000) return 0; // under one minute logs nothing
    const minutes = Math.round(ms / 60000);
    try {
      await api.focus(pomodoro.boundTaskId, minutes);
      return minutes;
    } catch (err) {
      actions.toast(`Could not log focus time: ${err.message}`, "error");
      return 0;
    }
  }

  async function creditPartial() {
    const ms = elapsed() - t.creditedMs;
    t.creditedMs = elapsed();
    const minutes = await credit(ms);
    if (minutes) message.textContent = `Logged ${minutes} focused minute${minutes === 1 ? "" : "s"} to the bound task.`;
  }

  function stopInterval() {
    if (t.interval) clearInterval(t.interval);
    t.interval = null;
  }

  function start() {
    t.runningSince = Date.now();
    stopInterval();
    t.interval = setInterval(tick, 250);
    message.textContent = "";
    paint();
  }

  function pause() {
    t.elapsedBefore = elapsed();
    t.runningSince = null;
    stopInterval();
    paint();
  }

  function resetCountdown() {
    stopInterval();
    t.elapsedBefore = 0;
    t.runningSince = null;
    t.creditedMs = 0;
  }

  async function reset() {
    pause();
    await creditPartial();
    resetCountdown();
    paint();
  }

  async function switchMode(key) {
    pause();
    await creditPartial();
    t.mode = key;
    resetCountdown();
    message.textContent = "";
    paint();
  }

  async function tick() {
    if (elapsed() < duration()) return paint();
    const wasFocus = t.mode === "focus";
    const uncredited = duration() - t.creditedMs;
    resetCountdown();
    paint();
    if (!wasFocus) {
      play("click");
      message.textContent = "Break over. Ready to focus?";
      return;
    }
    pomodoro.sessions += 1;
    play("fanfare");
    confetti();
    const minutes = await credit(uncredited);
    message.textContent = minutes ? `Focus session complete! ${minutes} minutes logged to the bound task.`
      : "Focus session complete! Bind a task to log focus time.";
    paint();
    actions.refresh();
  }

  const content = h("div", { class: "pomodoro" },
    modalHeader("Focus timer", () => close()),
    h("div", { class: "modal-body center" },
      modeButtons, display, h("div", { class: "progress" }, bar),
      h("div", { class: "timer-controls" }, startBtn, h("button", { class: "btn", onclick: () => reset() }, "Reset")),
      h("label", { class: "field" }, h("span", {}, "Log focus time to"), taskSelect),
      message, sessionCount));
  close = openModal(content, {
    className: "pomodoro-overlay", label: "Focus timer",
    onClose: () => { pause(); creditPartial().then(() => actions.refresh()); resetCountdown(); },
  });
  paint();

  // Bindable tasks: every active, incomplete task (not just the current filter).
  api.workspace({ scope: "active", sort: "custom" }).then((ws) => {
    const open = ws.tasks.filter((x) => x.status !== "completed");
    if (!open.some((x) => x.id === pomodoro.boundTaskId)) pomodoro.boundTaskId = null;
    clear(taskSelect, option("", "No task bound", pomodoro.boundTaskId), open.map((x) => option(x.id, x.title, pomodoro.boundTaskId)));
  }).catch((err) => actions.toast(err.message, "error"));
}
