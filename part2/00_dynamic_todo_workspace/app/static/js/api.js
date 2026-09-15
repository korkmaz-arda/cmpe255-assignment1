// Thin JSON client for the Python server (the single source of truth).

export class ApiError extends Error {}

async function request(method, url, body) {
  const options = { method, headers: {} };
  if (body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  let response;
  try {
    response = await fetch(url, options);
  } catch {
    throw new ApiError("Cannot reach the server");
  }
  if (response.status === 204) return null;
  let data = null;
  try {
    data = await response.json();
  } catch {
    /* non-JSON body */
  }
  if (!response.ok) throw new ApiError((data && data.error) || `Request failed (${response.status})`);
  return data;
}

const qs = (params) =>
  new URLSearchParams(Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "")).toString();

export const api = {
  workspace: (params) => request("GET", `/api/workspace?${qs(params)}`),
  analytics: () => request("GET", "/api/analytics"),
  calendar: (year, month) => request("GET", `/api/calendar?${qs({ year, month })}`),
  parse: (text) => request("POST", "/api/parse", { text }),
  capture: (payload) => request("POST", "/api/tasks", payload),
  getTask: (id) => request("GET", `/api/tasks/${id}`),
  updateTask: (id, data) => request("PUT", `/api/tasks/${id}`, data),
  patchTask: (id, data) => request("PATCH", `/api/tasks/${id}`, data),
  deleteTask: (id, permanent = false) => request("DELETE", `/api/tasks/${id}${permanent ? "?permanent=1" : ""}`),
  restore: (id) => request("POST", `/api/tasks/${id}/restore`),
  archive: (id) => request("POST", `/api/tasks/${id}/archive`),
  unarchive: (id) => request("POST", `/api/tasks/${id}/unarchive`),
  matrixDrop: (id, quadrant) => request("POST", `/api/tasks/${id}/matrix`, { quadrant }),
  focus: (id, minutes) => request("POST", `/api/tasks/${id}/focus`, { minutes }),
  reorder: (items, status) => request("POST", "/api/tasks/reorder", { items, status }),
  bulk: (action, ids, value) => request("POST", "/api/tasks/bulk", { action, ids, value }),
  createCategory: (name) => request("POST", "/api/categories", { name }),
  deleteCategory: (id) => request("DELETE", `/api/categories/${id}`),
  createTag: (name, color) => request("POST", "/api/tags", { name, color }),
  deleteTag: (id) => request("DELETE", `/api/tags/${id}`),
  activity: (limit) => request("GET", `/api/activity?${qs({ limit })}`),
  clearActivity: () => request("DELETE", "/api/activity"),
};
