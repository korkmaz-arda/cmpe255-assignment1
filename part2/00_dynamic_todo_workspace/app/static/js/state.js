// Session-only client state plus a small registry of cross-module actions.
// Everything under `server` is replaced wholesale by each refetch.

export const state = {
  view: "list", // list | board | matrix | calendar | analytics
  nav: { scope: "active", smart: "all", category: null },
  tag: null,
  search: "",
  sort: "custom",
  selected: new Set(),
  collapsedBuckets: new Set(),
  dragging: false,
  calendarMonth: null, // {year, month}; null = server's current month
  server: {
    today: null,
    tasks: [],
    counts: null,
    categories: [],
    tags: [],
    palettes: { category: [], tag: [], icons: [] },
    analytics: null,
    calendar: null,
  },
};

// Filled in by app.js so view/modal modules can trigger shared behavior
// without circular imports.
export const actions = {};

export const PRIORITY_LABELS = { urgent: "Urgent", high: "High", medium: "Medium", low: "Low" };
export const STATUS_LABELS = { todo: "To Do", in_progress: "In Progress", review: "In Review", completed: "Done" };

export const categoryById = (id) => state.server.categories.find((c) => c.id === id) || null;
export const taskById = (id) => state.server.tasks.find((t) => t.id === id) || null;
