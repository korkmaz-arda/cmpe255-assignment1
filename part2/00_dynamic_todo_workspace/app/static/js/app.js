// Application shell: data loading, live sync, sidebar, quick capture,
// selection/bulk actions, theming, and global keyboard shortcuts.

import { api } from "./api.js";
import { categoryIcon, clear, debounce, h, icon, isTypingTarget, safeStorage, saveStorage } from "./dom.js";
import { confetti, isMuted, play, setMuted, unlockAudio } from "./feedback.js";
import {
  closeTopOverlay, hasOverlay, openActivity, openEditor, openPomodoro, openTagManager, togglePalette, toggleShortcuts,
} from "./modals.js";
import { PRIORITY_LABELS, actions, state } from "./state.js";
import { closeDayComposer, renderAnalytics, renderBoard, renderCalendar, renderList, renderMatrix } from "./views.js";

const $ = (id) => document.getElementById(id);
const VIEWS = [
  ["list", "List", "list"], ["board", "Board", "board"], ["matrix", "Matrix", "matrix"],
  ["calendar", "Calendar", "calendar"], ["analytics", "Analytics", "chart"],
];
const SORTS = [
  ["custom", "Custom order"], ["due_asc", "Due date (soonest)"], ["due_desc", "Due date (latest)"],
  ["priority", "Priority (urgent → low)"], ["title", "Title A–Z"], ["recent", "Recently added"],
];
const ACCENTS = ["indigo", "cyan", "emerald", "amber", "rose"];

// ------------------------------------------------------------------ toast
function toast(text, kind = "info", action = null) {
  const el = h("div", { class: `toast ${kind}`, role: kind === "error" ? "alert" : "status" }, h("span", {}, text),
    action ? h("button", { class: "btn small", onclick: () => { action.run(); el.remove(); } }, action.label) : null);
  $("toast-root").append(el);
  setTimeout(() => el.remove(), action ? 6000 : 3500);
}

// ------------------------------------------------------------------- data
let refreshSeq = 0;
let refreshTimer = null;
let lastSnapshot = null;
let localEdits = false; // set when state.server was changed locally (optimistic update)

async function refresh({ fromSync = false } = {}) {
  const seq = ++refreshSeq;
  const cm = state.calendarMonth;
  try {
    const [workspace, analytics, calendar] = await Promise.all([
      api.workspace({
        scope: state.nav.scope, smart: state.nav.smart, category: state.nav.category, tag: state.tag,
        q: state.search, sort: state.sort,
      }),
      api.analytics(),
      api.calendar(cm && cm.year, cm && cm.month),
    ]);
    if (seq !== refreshSeq) return; // a newer refresh superseded this one
    const snapshot = JSON.stringify([workspace, analytics, calendar]);
    // Only broadcast-driven refetches may skip rendering; user-initiated ones changed client state.
    const unchanged = fromSync && snapshot === lastSnapshot && !localEdits;
    lastSnapshot = snapshot;
    localEdits = false;
    if (unchanged) return; // identical server state: avoid replacing the DOM (hover, drags, focus)
    Object.assign(state.server, workspace, { analytics, calendar });
    const visible = new Set(workspace.tasks.map((t) => t.id));
    for (const id of [...state.selected]) if (!visible.has(id)) state.selected.delete(id);
    if (state.tag !== null && !workspace.tags.some((t) => t.id === state.tag)) state.tag = null;
    if (state.nav.category !== null && !workspace.categories.some((c) => c.id === state.nav.category)) {
      state.nav = { scope: "active", smart: "all", category: null };
      return refresh();
    }
    render();
  } catch (err) {
    if (seq === refreshSeq) toast(err.message, "error");
  }
}

// Coalesce bursts of broadcast events into one refetch.
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  refreshTimer = setTimeout(() => refresh({ fromSync: true }), 60);
}

function connectLiveSync() {
  const status = $("sync-status");
  const source = new EventSource("/api/events");
  source.onmessage = (event) => {
    let message;
    try { message = JSON.parse(event.data); } catch { return; }
    if (message.type === "connected") {
      status.textContent = "Live";
      status.className = "sync-status live";
    }
    scheduleRefresh(); // uniform reaction: full refetch from the server
  };
  source.onerror = () => {
    status.textContent = "Reconnecting…";
    status.className = "sync-status offline";
    // EventSource reconnects automatically; the "connected" handshake triggers a refetch.
  };
}

// ------------------------------------------------------------- task actions
function replaceTask(task) {
  localEdits = true;
  const i = state.server.tasks.findIndex((t) => t.id === task.id);
  if (i !== -1) state.server.tasks[i] = task;
}

function completionFeedback() {
  play("complete");
  confetti();
}

async function toggleComplete(task) {
  const current = state.server.tasks.find((t) => t.id === task.id) || task;
  const completing = current.status !== "completed";
  const previous = { ...current };
  // Optimistic: flip the checkbox immediately; the server response replaces it.
  replaceTask({ ...current, status: completing ? "completed" : "todo" });
  render();
  if (completing) completionFeedback();
  else play("uncomplete");
  try {
    replaceTask(await api.patchTask(task.id, { status: completing ? "completed" : "todo" }));
  } catch (err) {
    replaceTask(previous);
    toast(`Could not update task: ${err.message}`, "error");
  }
  render();
  scheduleRefresh();
}

async function run(promise, success) {
  try {
    const result = await promise;
    if (success) success(result);
    await refresh();
    return true;
  } catch (err) {
    toast(err.message, "error");
    return false;
  }
}

async function deleteTask(task) {
  if (task.deleted) {
    if (!confirm(`Permanently delete “${task.title}”? This cannot be undone.`)) return false;
    return run(api.deleteTask(task.id), () => { play("delete"); toast("Task permanently deleted"); });
  }
  return run(api.deleteTask(task.id), () => {
    play("delete");
    toast("Moved to trash", "info", { label: "Undo", run: () => run(api.restore(task.id)) });
  });
}

async function capture(payload) {
  const task = await api.capture(payload);
  play("click");
  if (task.status === "completed") completionFeedback();
  await refresh();
  return task;
}

Object.assign(actions, {
  refresh, render, toast,
  dragEnded: () => { if (viewRenderPending) setTimeout(render, 0); }, completionFeedback, toggleComplete, deleteTask, capture, openEditor,
  togglePin: (task) => run(api.patchTask(task.id, { pinned: !task.pinned })),
  restore: (task) => run(api.restore(task.id), () => toast("Task restored")),
  archive: (task) => run(api.archive(task.id), () => toast("Task archived")),
  unarchive: (task) => run(api.unarchive(task.id), () => toast("Task returned from archive")),
  toggleSelect: (id) => { state.selected.has(id) ? state.selected.delete(id) : state.selected.add(id); render(); },
  selectAll: (on) => { state.selected = on ? new Set(state.server.tasks.map((t) => t.id)) : new Set(); render(); },
  setView,
  setNav,
  toggleTheme,
  setAccent,
});

// -------------------------------------------------------------- navigation
function setView(view) {
  state.view = view;
  render();
  refresh(); // re-derive from the server (e.g. analytics, date rollover) when a view is opened
}

function setNav(nav) {
  state.nav = { scope: "active", smart: "all", category: null, ...nav };
  state.selected.clear();
  refresh();
}

// ------------------------------------------------------------------ theme
function applyPrefs() {
  document.documentElement.dataset.theme = safeStorage("zenith.theme", "dark");
  document.documentElement.dataset.accent = safeStorage("zenith.accent", "indigo");
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  saveStorage("zenith.theme", next);
  applyPrefs();
  render();
}

function setAccent(accent) {
  saveStorage("zenith.accent", ACCENTS.includes(accent) ? accent : "indigo");
  applyPrefs();
  render();
}

// ---------------------------------------------------------------- render
let viewRenderPending = false;

function render() {
  renderSidebar();
  renderTopbar();
  renderScopeBanner();
  renderCaptureCategories();
  renderBulkBar();
  if (state.dragging) {
    // Replacing the DOM mid-drag would cancel the gesture; render when it ends.
    viewRenderPending = true;
    return;
  }
  viewRenderPending = false;
  const root = $("view-root");
  // Keep focus and typed text in inputs inside the view (e.g. board quick-add).
  const active = document.activeElement;
  const label = active && root.contains(active) && active.tagName === "INPUT" && !active.closest(".day-composer")
    ? active.getAttribute("aria-label") : null;
  const saved = label ? { value: active.value, start: active.selectionStart, end: active.selectionEnd } : null;
  if (state.view !== "calendar") closeDayComposer();
  root.dataset.view = state.view;
  ({ list: renderList, board: renderBoard, matrix: renderMatrix, calendar: renderCalendar, analytics: renderAnalytics })[state.view](root);
  if (label) {
    const again = [...root.querySelectorAll("input")].find((el) => el.getAttribute("aria-label") === label);
    if (again) {
      again.value = saved.value;
      again.focus();
      try { again.setSelectionRange(saved.start, saved.end); } catch { /* not a text input */ }
    }
  }
}

function navItem({ label, iconNode, count, active, onclick, title, testid }) {
  return h("button", { class: `nav-item${active ? " active" : ""}`, onclick, title: title || label, "aria-current": active ? "page" : null, "data-nav": testid },
    iconNode, h("span", { class: "label" }, label), count !== undefined ? h("span", { class: "count" }, count) : null);
}

function renderSidebar() {
  const counts = state.server.counts;
  const smart = counts ? counts.smart : {};
  const { scope, smart: activeSmart, category } = state.nav;
  clear($("smart-nav"), [
    ["all", "All Tasks", "inbox"], ["today", "Today", "sun"], ["upcoming", "Upcoming", "upcoming"],
    ["important", "Important", "star"], ["completed", "Completed", "checkCircle"],
  ].map(([key, label, ic]) => navItem({
    label, iconNode: icon(ic), count: smart[key] ?? 0, testid: `smart-${key}`,
    active: scope === "active" && category === null && activeSmart === key,
    onclick: () => setNav({ smart: key }),
  })));

  clear($("category-nav"), state.server.categories.map((c) => h("div", { class: "nav-row" },
    navItem({
      label: c.name, iconNode: h("span", { class: "cat-icon", style: { color: c.color } }, categoryIcon(c.icon)),
      count: (counts && counts.categories[c.id]) || 0, active: scope === "active" && category === c.id,
      onclick: () => setNav({ category: c.id }), testid: `category-${c.id}`,
    }),
    h("button", { class: "icon-btn nav-delete", "aria-label": `Delete category ${c.name}`, title: "Delete category (tasks are kept)",
      onclick: () => {
        if (!confirm(`Delete category “${c.name}”? Its tasks are kept and become uncategorized.`)) return;
        run(api.deleteCategory(c.id), () => toast("Category deleted"));
      } }, icon("x", 12)))));

  clear($("tag-nav"), state.server.tags.length ? state.server.tags.map((t) => h("button", {
    class: `chip chip-tag toggle${state.tag === t.id ? " on" : ""}`, style: { "--chip": t.color }, "aria-pressed": String(state.tag === t.id),
    title: `${(counts && counts.tags[t.id]) || 0} active tasks`,
    onclick: () => { state.tag = state.tag === t.id ? null : t.id; state.selected.clear(); refresh(); },
  }, `#${t.name}`)) : h("span", { class: "muted small" }, "No tags"));

  clear($("lifecycle-nav"),
    navItem({ label: "Archive", iconNode: icon("archive"), count: counts ? counts.archive : 0, active: scope === "archive", onclick: () => setNav({ scope: "archive" }), testid: "archive" }),
    navItem({ label: "Trash", iconNode: icon("trash"), count: counts ? counts.trash : 0, active: scope === "trash", onclick: () => setNav({ scope: "trash" }), testid: "trash" }));

  const mute = $("mute-toggle");
  clear(mute, icon(isMuted() ? "mute" : "volume"), h("span", { class: "label" }, isMuted() ? "Sound off" : "Sound on"));
  mute.setAttribute("aria-pressed", String(isMuted()));
  const theme = $("theme-toggle");
  const dark = document.documentElement.dataset.theme === "dark";
  clear(theme, icon(dark ? "sun" : "moon"), h("span", { class: "label" }, dark ? "Light mode" : "Dark mode"));
  clear($("accent-picker"), ACCENTS.map((a) => h("button", {
    class: `accent-dot accent-${a}${document.documentElement.dataset.accent === a ? " on" : ""}`, title: `Accent: ${a}`, "aria-label": `Accent ${a}`,
    onclick: () => setAccent(a),
  })));
}

function renderTopbar() {
  clear($("view-switcher"), VIEWS.map(([key, label, ic]) => h("button", {
    class: `view-tab${state.view === key ? " active" : ""}`, "data-view": key, "aria-pressed": String(state.view === key),
    onclick: () => setView(key),
  }, icon(ic), h("span", {}, label))));
  const sort = $("sort-select");
  if (!sort.options.length) SORTS.forEach(([v, l]) => sort.append(h("option", { value: v }, l)));
  sort.value = state.sort;
}

function renderScopeBanner() {
  const banner = $("scope-banner");
  const parts = [];
  if (state.nav.scope === "trash") parts.push("Trash: tasks here are hidden from all other views and from analytics. Deleting again removes them permanently.");
  if (state.nav.scope === "archive") parts.push("Archive: archived tasks are hidden from normal views and analytics.");
  const cat = state.server.categories.find((c) => c.id === state.nav.category);
  if (cat) parts.push(`Category: ${cat.name}`);
  const tag = state.server.tags.find((t) => t.id === state.tag);
  if (tag) parts.push(`Tag: #${tag.name}`);
  if (state.search) parts.push(`Search: “${state.search}”`);
  banner.hidden = parts.length === 0;
  clear(banner, parts.join(" · "));
}

function renderCaptureCategories() {
  const select = $("capture-category");
  const value = select.value;
  clear(select, h("option", { value: "" }, "Category…"), state.server.categories.map((c) => h("option", { value: c.id }, c.name)));
  select.value = state.server.categories.some((c) => String(c.id) === value) ? value : "";
  const disabled = state.nav.scope !== "active";
  $("capture-input").disabled = disabled;
  $("capture-input").placeholder = disabled ? "Switch to an active list to capture tasks"
    : "Add a task… e.g. Review PR !high #backend ~30m @tomorrow";
}

function renderBulkBar() {
  const bar = $("bulk-bar");
  const ids = [...state.selected];
  bar.hidden = ids.length === 0;
  if (!ids.length) return clear(bar);
  const bulk = (action, value, message) => run(api.bulk(action, ids, value), () => {
    if (action === "complete") completionFeedback();
    if (action === "trash" || action === "purge") play("delete");
    toast(message);
    if (["trash", "purge", "restore", "archive", "unarchive"].includes(action)) state.selected.clear();
  });
  const btn = (label, ic, onclick, cls = "") => h("button", { class: `btn ${cls}`, onclick }, icon(ic, 14), label);
  const scope = state.nav.scope;
  const controls = scope === "trash" ? [
    btn("Restore", "restore", () => bulk("restore", null, "Tasks restored")),
    btn("Delete forever", "trash", () => { if (confirm(`Permanently delete ${ids.length} task(s)?`)) bulk("purge", null, "Tasks permanently deleted"); }, "danger"),
  ] : [
    btn("Complete", "check", () => bulk("complete", null, "Marked complete")),
    btn("Reopen", "restore", () => bulk("uncomplete", null, "Marked incomplete")),
    h("select", { class: "input compact", "aria-label": "Set priority for selected", onchange: (e) => e.target.value && bulk("priority", e.target.value, "Priority updated") },
      h("option", { value: "" }, "Priority…"), Object.entries(PRIORITY_LABELS).map(([v, l]) => h("option", { value: v }, l))),
    h("select", { class: "input compact", "aria-label": "Set category for selected",
      onchange: (e) => e.target.value !== "" && bulk("category", e.target.value === "none" ? null : Number(e.target.value), "Category updated") },
    h("option", { value: "" }, "Category…"), h("option", { value: "none" }, "No category"),
    state.server.categories.map((c) => h("option", { value: c.id }, c.name))),
    scope === "archive" ? btn("Unarchive", "restore", () => bulk("unarchive", null, "Returned from archive"))
      : btn("Archive", "archive", () => bulk("archive", null, "Archived")),
    btn("Trash", "trash", () => bulk("trash", null, "Moved to trash"), "danger"),
  ];
  clear(bar, h("span", { class: "bulk-count" }, `${ids.length} selected`), controls,
    h("button", { class: "icon-btn", "aria-label": "Clear selection", title: "Clear selection", onclick: () => { state.selected.clear(); render(); } }, icon("x")));
}

// ---------------------------------------------------------- quick capture
function setupCapture() {
  const input = $("capture-input");
  const chips = $("capture-chips");
  const submit = $("capture-submit");
  let previewText = null;
  let previewTitle = "";

  const updateSubmit = () => {
    // Lightweight UX check only; the server re-parses and validates on submit.
    const text = input.value;
    submit.disabled = !text.trim() || (previewText === text && !previewTitle);
  };

  const preview = debounce(async () => {
    const text = input.value;
    if (!text.trim()) {
      previewText = text;
      previewTitle = "";
      clear(chips);
      return updateSubmit();
    }
    try {
      const parsed = await api.parse(text);
      if (input.value !== text) return; // stale
      previewText = text;
      previewTitle = parsed.title;
      clear(chips, parsed.title ? h("span", { class: "chip chip-title" }, `Title: ${parsed.title}`) : h("span", { class: "chip warn" }, "Title is empty"),
        parsed.chips.map((c) => h("span", { class: `chip token token-${c.kind}${c.recognized ? "" : " unrecognized"}`, "data-kind": c.kind },
          h("b", {}, { priority: "Priority", tag: "Tag", estimate: "Estimate", due: "Due" }[c.kind]), " ", c.label)));
      updateSubmit();
    } catch {
      /* preview is best-effort; submit still validates on the server */
    }
  }, 120);

  input.addEventListener("input", () => { updateSubmit(); preview(); });
  $("capture-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    unlockAudio();
    if (submit.disabled) return;
    const categoryValue = $("capture-category").value;
    const categoryId = categoryValue ? Number(categoryValue) : state.nav.category;
    try {
      await capture({ capture: input.value, category_id: categoryId });
      input.value = "";
      previewText = "";
      previewTitle = "";
      clear(chips);
      updateSubmit();
    } catch (err) {
      toast(err.message, "error");
    }
  });
  updateSubmit();
}

// -------------------------------------------------------------- shortcuts
function setupShortcuts() {
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
      e.preventDefault();
      togglePalette();
      return;
    }
    if (e.key === "Escape") {
      if (closeTopOverlay()) e.preventDefault();
      else closeDayComposer();
      return;
    }
    if (isTypingTarget(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") {
      if (hasOverlay()) return;
      e.preventDefault();
      $("search-input").focus();
    } else if (e.key === "?") {
      e.preventDefault();
      toggleShortcuts();
    }
  });
}

// ------------------------------------------------------------------ init
function init() {
  applyPrefs();
  document.addEventListener("pointerdown", unlockAudio, { once: true });
  document.addEventListener("keydown", unlockAudio, { once: true });

  $("sidebar-toggle").addEventListener("click", () => {
    const collapsed = $("app").classList.toggle("sidebar-collapsed");
    $("sidebar-toggle").setAttribute("aria-expanded", String(!collapsed));
  });
  $("mute-toggle").addEventListener("click", () => { setMuted(!isMuted()); render(); });
  $("theme-toggle").addEventListener("click", toggleTheme);
  $("tags-button").addEventListener("click", openTagManager);
  $("activity-button").addEventListener("click", openActivity);
  $("shortcuts-button").addEventListener("click", toggleShortcuts);
  $("palette-button").addEventListener("click", togglePalette);
  $("pomodoro-button").addEventListener("click", () => { unlockAudio(); openPomodoro(); });

  const categoryForm = $("category-form");
  $("add-category").addEventListener("click", () => {
    categoryForm.hidden = !categoryForm.hidden;
    if (!categoryForm.hidden) categoryForm.querySelector("input").focus();
  });
  categoryForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const input = categoryForm.querySelector("input");
    if (!input.value.trim()) return;
    if (await run(api.createCategory(input.value), () => toast("Category created"))) {
      input.value = "";
      categoryForm.hidden = true;
    }
  });
  categoryForm.querySelector("input").addEventListener("keydown", (e) => {
    if (e.key === "Escape") { e.stopPropagation(); categoryForm.hidden = true; }
  });

  const search = $("search-input");
  search.addEventListener("input", debounce(() => { state.search = search.value; refresh(); }, 150));
  search.addEventListener("keydown", (e) => { if (e.key === "Escape") { search.value = ""; state.search = ""; search.blur(); refresh(); } });
  $("sort-select").addEventListener("change", (e) => { state.sort = e.target.value; refresh(); });

  setupCapture();
  setupShortcuts();
  render();
  refresh();
  connectLiveSync();
}

init();
