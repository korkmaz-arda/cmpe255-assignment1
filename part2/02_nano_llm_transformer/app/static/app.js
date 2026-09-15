"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const el = (tag, attrs = {}, ...children) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c);
  return node;
};
const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const fmtInt = (v) => (v == null ? "–" : Number(v).toLocaleString("en-US"));
const glyph = (t) => (t === " " ? "␣" : t === "\n" ? "↵" : t === "\t" ? "⇥" : t === "\r" ? "␍" : t);
// Version ids start with a creation timestamp (YYYYMMDD-HHMMSS-); the UI shows only the run name.
const shortVersion = (v) => (v ? String(v).replace(/^\d{8}-\d{6}-/, "") : v);
const SOURCE_LABELS = { tinystories: "TinyStories", everyday: "Everyday Conversations", kb: "Knowledge base", kb_rephrase: "KB rephrasing diagnostic" };

async function api(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* ignore */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

/* ------------------------------------------------------------------ tooltips */
const tooltip = $("#tooltip");
document.addEventListener("mouseover", (e) => {
  const t = e.target.closest("[data-tip]");
  if (!t) return;
  tooltip.textContent = t.dataset.tip;
  tooltip.classList.remove("hidden");
});
document.addEventListener("mousemove", (e) => {
  if (tooltip.classList.contains("hidden")) return;
  const x = Math.min(e.clientX + 14, window.innerWidth - 340);
  tooltip.style.left = `${Math.max(8, x)}px`;
  tooltip.style.top = `${e.clientY + 14}px`;
});
document.addEventListener("mouseout", (e) => {
  if (e.target.closest("[data-tip]")) tooltip.classList.add("hidden");
});

/* ------------------------------------------------------------------ tabs (S02) */
const loaded = {};
function showTab(name) {
  $$(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  if (!loaded[name]) {
    loaded[name] = true;
    ({ attention: loadAttention, tokenizer: loadTokenizer, telemetry: loadTelemetry, architecture: loadArchitecture }[name] || (() => {}))();
  }
}
$$(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

/* ------------------------------------------------------------------ health (S06) */
let health = null;
async function refreshHealth() {
  const pill = $("#health");
  const banner = $("#global-banner");
  try {
    health = await api("/api/health");
    if (health.model_loaded) {
      pill.textContent = `${shortVersion(health.version)} · ${fmtInt(health.parameters)} params · vocab ${health.vocab_size} · ${health.device}`;
      pill.className = "health-pill mono ok";
      banner.classList.add("hidden");
    } else {
      pill.textContent = "no model loaded";
      pill.className = "health-pill mono bad";
      banner.textContent = health.error || "No model is loaded.";
      banner.classList.remove("hidden");
    }
  } catch (err) {
    pill.textContent = "server unreachable";
    pill.className = "health-pill mono bad";
  }
}

/* ------------------------------------------------------------------ chat (F06) */
const sliders = ["temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"];
sliders.forEach((id) => {
  const input = $(`#${id}`);
  const out = $(`#${id}-out`);
  const update = () => { out.textContent = input.value; };
  input.addEventListener("input", update);
  update();
});

// Initialize decoding controls from the server's canonical defaults (shared with probes and finalization).
async function loadDecodingDefaults() {
  try {
    const d = (await api("/api/decoding-defaults")).default;
    sliders.forEach((id) => { $(`#${id}`).value = d[id]; $(`#${id}-out`).textContent = $(`#${id}`).value; });
    $("#loop_guard").checked = d.loop_guard;
    $("#seed").value = d.seed == null ? "" : d.seed;
  } catch (_) { /* HTML values already mirror the defaults */ }
}

let personas = [];
let history = []; // [{role, content}]
let streaming = false;

const GREETING = "Hi! I'm NanoLlama, a tiny character-level transformer (RoPE, SwiGLU, RMSNorm, KV cache) trained from scratch on synthetic stories, synthetic small-talk chats and a small knowledge base. Try a preset on the left or type a message. My replies are generated live and are often wrong.";

function addMessage(role, text, opts = {}) {
  const msg = el("div", { class: `msg ${role}${opts.static ? " static" : ""}${opts.error ? " error" : ""}` });
  const avatar = el("div", { class: "avatar", text: role === "user" ? "U" : "N" });
  const bubble = el("div", { class: "bubble", text });
  const wrap = el("div", {}, bubble);
  if (opts.meta) wrap.append(el("div", { class: "meta", text: opts.meta }));
  msg.append(avatar, wrap);
  $("#transcript").append(msg);
  $("#transcript").scrollTop = $("#transcript").scrollHeight;
  return { msg, bubble, wrap };
}

function resetChat() {
  history = [];
  $("#transcript").innerHTML = "";
  addMessage("assistant", GREETING, { static: true, meta: "Welcome note — static, not model-generated" });
  ["m-tps", "m-ttft", "m-count", "m-total", "m-cache"].forEach((id) => { $(`#${id}`).textContent = "–"; });
}

async function loadPresets() {
  try {
    const data = await api("/api/presets");
    personas = data.personas;
    const sel = $("#persona");
    sel.innerHTML = "";
    personas.forEach((p) => sel.append(el("option", { value: p.key, text: p.label })));
    sel.append(el("option", { value: "custom", text: "Custom (out of distribution)" }));
    sel.value = "assistant";
    $("#system").value = personas.find((p) => p.key === "assistant").system;
    const box = $("#presets");
    box.innerHTML = "";
    data.presets.forEach((p) => {
      const b = el("button", { class: "preset", type: "button" },
        el("div", { class: "cat", text: p.category }),
        el("div", { text: p.title.split("\n")[0] }),
        el("div", { class: "origin", text: p.origin }));
      b.addEventListener("click", () => {
        $("#persona").value = p.persona;
        $("#system").value = p.system;
        sendMessage(p.message);
      });
      box.append(b);
    });
  } catch (err) {
    $("#presets").textContent = `Could not load presets: ${err.message}`;
  }
}
$("#persona").addEventListener("change", (e) => {
  const p = personas.find((x) => x.key === e.target.value);
  if (p) $("#system").value = p.system;
});
$("#system").addEventListener("input", () => {
  const p = personas.find((x) => x.system === $("#system").value);
  $("#persona").value = p ? p.key : "custom";
});

async function sendMessage(text) {
  text = (text ?? $("#message").value).trim();
  if (!text || streaming) return;
  streaming = true;
  $("#send").disabled = true;
  $("#message").value = "";
  addMessage("user", text);
  const { bubble, wrap } = addMessage("assistant", "");
  bubble.classList.add("cursor");
  const seedVal = $("#seed").value.trim();
  const body = {
    message: text,
    system: $("#system").value,
    history,
    temperature: parseFloat($("#temperature").value),
    top_p: parseFloat($("#top_p").value),
    top_k: parseInt($("#top_k").value, 10),
    repetition_penalty: parseFloat($("#repetition_penalty").value),
    max_new_tokens: parseInt($("#max_new_tokens").value, 10),
    loop_guard: $("#loop_guard").checked,
    seed: seedVal === "" ? null : parseInt(seedVal, 10),
  };
  let reply = "";
  let failed = null;
  let done = null;
  let start = null;
  try {
    const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail; } catch (_) { /* ignore */ }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      buf += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === "start") start = ev;
        else if (ev.type === "token") {
          reply += ev.text;
          bubble.textContent = reply;
          $("#m-tps").textContent = ev.tokens_per_second ? ev.tokens_per_second.toFixed(0) : "…";
          $("#m-ttft").textContent = `${ev.ttft_ms.toFixed(0)} ms`;
          $("#m-count").textContent = ev.count;
          $("#m-total").textContent = `${(ev.elapsed_ms / 1000).toFixed(2)} s`;
          $("#transcript").scrollTop = $("#transcript").scrollHeight;
        } else if (ev.type === "done") {
          done = ev;
        } else if (ev.type === "error") {
          failed = ev.message;
        }
      }
    }
  } catch (err) {
    failed = err.message;
  }
  bubble.classList.remove("cursor");
  if (done) {
    $("#m-tps").textContent = done.tokens_per_second ? done.tokens_per_second.toFixed(0) : "–";
    $("#m-ttft").textContent = `${done.ttft_ms.toFixed(0)} ms`;
    $("#m-count").textContent = done.count;
    $("#m-total").textContent = `${(done.total_ms / 1000).toFixed(2)} s`;
    $("#m-cache").textContent = done.kv_cache ? `on (${done.device})` : "off";
    const bits = [`${done.count} chars`, `stop: ${done.stop_reason}`];
    if (start && start.history_turns_dropped) bits.push(`${start.history_turns_dropped} old turns dropped to fit context`);
    if (!reply) bits.unshift("empty reply: the model emitted end-of-sequence immediately (no text shown is model output)");
    wrap.append(el("div", { class: "meta", text: bits.join(" · ") }));
    history.push({ role: "user", content: text }, { role: "assistant", content: reply });
  }
  if (failed) {
    bubble.parentElement.parentElement.classList.add("error");
    bubble.textContent = (reply ? `${reply}\n\n` : "") + `Generation failed: ${failed}`;
  }
  streaming = false;
  $("#send").disabled = false;
}
$("#chat-form").addEventListener("submit", (e) => { e.preventDefault(); sendMessage(); });
$("#message").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
$("#reset-chat").addEventListener("click", resetChat);

/* ------------------------------------------------------------------ attention (F07) */
let attnData = null;
let attnLayer = 0;
const RAMP = [[17, 24, 39], [24, 79, 149], [57, 135, 229], [134, 182, 239], [205, 226, 251]];
function rampColor(v) {
  const x = Math.max(0, Math.min(1, v)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x));
  const f = x - i;
  const c = RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
const tokLabel = (t) => (t === "<|bos|>" ? "BOS" : glyph(t));

function setLoading(btn, on) {
  btn.disabled = on;
  btn.classList.toggle("loading", on);
}

async function loadAttention(e) {
  if (e) e.preventDefault();
  const btn = $("#attn-btn");
  const err = $("#attn-error");
  setLoading(btn, true);
  err.classList.add("hidden");
  try {
    attnData = await api("/api/attention", { method: "POST", body: JSON.stringify({ text: $("#attn-text").value }) });
    attnLayer = Math.min(attnLayer, attnData.n_layers - 1);
    renderLayerButtons();
    renderAttention();
  } catch (ex) {
    err.textContent = `Could not compute attention: ${ex.message}`;
    err.classList.remove("hidden");
  } finally {
    setLoading(btn, false);
  }
}
$("#attn-form").addEventListener("submit", loadAttention);

function renderLayerButtons() {
  const box = $("#layer-buttons");
  box.innerHTML = "";
  for (let l = 0; l < attnData.n_layers; l++) {
    const b = el("button", { type: "button", class: l === attnLayer ? "active" : "", text: `Layer ${l + 1}` });
    b.addEventListener("click", () => { attnLayer = l; renderLayerButtons(); renderAttention(); });
    box.append(b);
  }
}

function renderAttention() {
  const grids = $("#attn-grids");
  grids.innerHTML = "";
  const n = attnData.size;
  const labelW = 26;
  const cell = Math.max(4, Math.floor(300 / n));
  const size = labelW + cell * n;
  const dpr = window.devicePixelRatio || 1;
  attnData.attention[attnLayer].forEach((head, h) => {
    const canvas = el("canvas");
    const height = cell * n;
    canvas.width = size * dpr;
    canvas.height = height * dpr;
    canvas.style.maxWidth = `${size}px`;
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);
    ctx.fillStyle = "#0b0f18";
    ctx.fillRect(0, 0, size, height);
    for (let q = 0; q < n; q++) {
      for (let k = 0; k < n; k++) {
        const x = labelW + k * cell;
        const y = q * cell;
        if (k > q) {
          ctx.fillStyle = "#1a2233";
          ctx.fillRect(x, y, cell - (cell > 5 ? 1 : 0), cell - (cell > 5 ? 1 : 0));
          if (cell >= 6) {
            ctx.strokeStyle = "#0b0f18";
            ctx.beginPath(); ctx.moveTo(x, y + cell); ctx.lineTo(x + cell, y); ctx.stroke();
          }
        } else {
          ctx.fillStyle = rampColor(head[q][k]);
          ctx.fillRect(x, y, cell - (cell > 5 ? 1 : 0), cell - (cell > 5 ? 1 : 0));
        }
      }
      if (cell >= 8) {
        ctx.fillStyle = "#a9b6cb";
        ctx.font = `${Math.min(11, cell)}px ui-monospace, monospace`;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(tokLabel(attnData.tokens[q]).slice(0, 3), labelW - 3, q * cell + cell / 2);
      }
    }
    canvas.addEventListener("mousemove", (ev) => {
      const r = canvas.getBoundingClientRect();
      const scale = size / r.width;
      const k = Math.floor(((ev.clientX - r.left) * scale - labelW) / cell);
      const q = Math.floor(((ev.clientY - r.top) * scale) / cell);
      if (q < 0 || k < 0 || q >= n || k >= n) return;
      const banner = $("#attn-hover");
      if (k > q) {
        banner.innerHTML = `Layer ${attnLayer + 1}, head ${h + 1}: query <b>“${tokLabel(attnData.tokens[q])}”</b> (#${q}) cannot attend to future key <b>“${tokLabel(attnData.tokens[k])}”</b> (#${k}) — masked`;
      } else {
        banner.innerHTML = `Layer ${attnLayer + 1}, head ${h + 1}: query <b>“${tokLabel(attnData.tokens[q])}”</b> (#${q}) attends to key <b>“${tokLabel(attnData.tokens[k])}”</b> (#${k}) with weight <b>${(100 * head[q][k]).toFixed(2)}%</b>`;
      }
    });
    const card = el("div", { class: "head-card" },
      el("div", { class: "cap mono" }, el("span", { text: `Head ${h + 1}` }), el("span", { text: `${n} × ${n}` })),
      canvas);
    grids.append(card);
  });
  if (attnData.truncated) {
    grids.prepend(el("div", { class: "muted small", text: `Prompt truncated to the first ${attnData.max_tokens} tokens (including BOS).` }));
  }
}

/* ------------------------------------------------------------------ tokenizer (F08) */
let tokTimer = null;
let tokSeq = 0;
async function loadTokenizer() {
  try {
    const rows = await api("/api/control-tokens");
    const tbody = $("#control-table tbody");
    tbody.innerHTML = "";
    rows.forEach((r) => tbody.append(el("tr", {}, el("td", { text: r.id }), el("td", { text: r.token }), el("td", { text: r.description }))));
  } catch (err) { /* table stays empty; tokenize shows the error */ }
  tokenize();
}
async function tokenize() {
  const seq = ++tokSeq;
  const err = $("#tok-error");
  try {
    const d = await api("/api/tokenize", { method: "POST", body: JSON.stringify({ text: $("#tok-text").value }) });
    if (seq !== tokSeq) return;
    err.classList.add("hidden");
    $("#tok-count").textContent = d.total_tokens;
    $("#tok-chars").textContent = d.characters;
    $("#tok-ratio").textContent = d.chars_per_token == null ? "–" : d.chars_per_token.toFixed(2);
    const chips = $("#tok-chips");
    chips.innerHTML = "";
    d.tokens.forEach((t, i) => chips.append(el("span", { class: `chip c${i % 5}`, title: `id ${t.id}` },
      el("span", { class: "g", text: glyph(t.text) }), el("span", { class: "i", text: t.id }))));
    const notes = [];
    if (d.normalized) notes.push("Typographic characters were normalized to ASCII before tokenizing (e.g. “…” → \"...\"), so token and character counts can differ.");
    if (d.replaced_with_space) notes.push(`${d.replaced_with_space} character(s) outside the vocabulary were mapped to a space.`);
    $("#tok-note").textContent = notes.join(" ");
    const preds = $("#tok-preds");
    preds.innerHTML = "";
    d.top_predictions.forEach((p) => preds.append(el("div", { class: "pred" },
      el("span", { class: "glyph", text: p.is_control ? p.text.replace("<|", "").replace("|>", "") : glyph(p.text), title: `id ${p.id}` }),
      el("div", {}, el("div", { class: "bar" }, el("div", { style: `width:${p.percent.toFixed(1)}%` })), el("div", { class: "muted small", text: `id ${p.id}` })),
      el("span", { class: "pct", text: `${p.percent.toFixed(1)}%` }))));
  } catch (ex) {
    if (seq !== tokSeq) return;
    err.textContent = `Tokenizer unavailable: ${ex.message}`;
    err.classList.remove("hidden");
  }
}
$("#tok-text").addEventListener("input", () => { clearTimeout(tokTimer); tokTimer = setTimeout(tokenize, 120); });

/* ------------------------------------------------------------------ telemetry (F09) */
const SERIES = [
  { key: "train", label: "Train (mixture)", color: "var(--series-1)" },
  { key: "tinystories", label: "Val · TinyStories", color: "var(--series-2)" },
  { key: "everyday", label: "Val · Everyday", color: "var(--series-3)" },
  { key: "kb", label: "Val · KB (diagnostic, not used for selection)", color: "var(--series-4)" },
  { key: "selection", label: "Selection loss = (TinyStories + Everyday) / 2", color: "var(--series-5)", dashed: true },
];

async function loadTelemetry() {
  let t;
  const empty = $("#tel-empty");
  const content = $("#tel-content");
  try {
    t = await api("/api/telemetry");
  } catch (err) {
    empty.innerHTML = `<div class="banner banner-error">Could not load telemetry: ${err.message}</div>`;
    empty.classList.remove("hidden");
    content.classList.add("hidden");
    return;
  }
  if (!t.available) {
    empty.innerHTML = "";
    empty.append(el("div", { class: "empty-state" },
      el("h2", { text: "No training run yet" }),
      el("p", { text: t.reason || "No promoted model version was found." }),
      el("p", { html: "Train one with <code>python scripts/train.py</code>. No placeholder numbers are shown here." })));
    empty.classList.remove("hidden");
    content.classList.add("hidden");
    return;
  }
  empty.classList.add("hidden");
  content.classList.remove("hidden");
  const c = t.model_config;
  const tr = t.training;
  $("#tel-version").textContent = `version ${shortVersion(t.version)} · ${tr.environment.gpu || tr.environment.device}`;
  $("#tel-lede").textContent = `Trained from ${tr.training_from}. Objective: ${tr.objective}. ${tr.epochs} epochs, batch ${tr.batch_size}, peak learning rate ${tr.learning_rate}, ${tr.optimizer.name} (betas ${tr.optimizer.betas.join("/")}, weight decay ${tr.optimizer.weight_decay}), ${tr.optimizer.schedule}, gradient clipping at ${tr.optimizer.grad_clip_norm}. Wall clock ${(tr.wall_clock_seconds / 60).toFixed(1)} min, ${fmtInt(Math.round(tr.mean_training_tokens_per_second))} training tokens/s (measured).`;

  const sel = t.selected_validation;
  const kpis = [
    { label: "Parameters", value: fmtInt(t.parameters), cap: `${c.n_layers} layers · ${c.n_heads} heads · width ${c.d_model} · tied embeddings` },
    { label: "Validation selection loss", value: fmt(sel.selection_loss, 4), cap: `cross-entropy, nats/char · (TinyStories + Everyday) / 2 at saved epoch ${tr.best_epoch}`, tip: tr.selection_definition },
    { label: "Validation perplexity", value: fmt(sel.selection_perplexity, 3), cap: "exp(selection loss)", tip: tr.perplexity_definition },
    { label: "Context window", value: fmtInt(c.context_length), cap: "characters (tokens)" },
  ];
  const box = $("#kpis");
  box.innerHTML = "";
  kpis.forEach((k) => box.append(el("div", { class: "kpi" },
    el("div", { class: "label" }, k.label, k.tip ? el("span", { class: "tip", "data-tip": k.tip, text: "?", style: "margin-left:6px" }) : null),
    el("div", { class: "value", text: k.value }),
    el("div", { class: "cap", text: k.cap }))));

  renderLossChart(t.curves, tr.best_epoch);
  renderValidationTables(t);
  renderSamples(t.validation);
  renderFinal(t);
}

function lossSeries(curves) {
  return {
    train: curves.train_loss,
    tinystories: curves.val_loss.tinystories,
    everyday: curves.val_loss.everyday,
    kb: curves.val_loss.kb,
    selection: curves.val_selection,
  };
}

function renderLossChart(curves, bestEpoch) {
  const data = lossSeries(curves);
  const n = data.train.length;
  const W = 900, H = 300, m = { l: 48, r: 16, t: 12, b: 36 };
  const all = Object.values(data).flat();
  const yMax = Math.max(...all) * 1.05;
  const yMin = Math.max(0, Math.min(...all) * 0.9);
  const x = (i) => m.l + (n === 1 ? (W - m.l - m.r) / 2 : (i * (W - m.l - m.r)) / (n - 1));
  const y = (v) => m.t + (1 - (v - yMin) / (yMax - yMin)) * (H - m.t - m.b);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Training and validation loss per epoch");
  const add = (tag, attrs, parent = svg) => {
    const e = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([k, v]) => e.setAttribute(k, v));
    parent.append(e);
    return e;
  };
  const grid = add("g", { class: "grid" });
  const axis = add("g", { class: "axis" });
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = yMin + ((yMax - yMin) * i) / ticks;
    add("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }, grid);
    const tx = add("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, axis);
    tx.textContent = v.toFixed(2);
  }
  const step = Math.max(1, Math.ceil(n / 12));
  for (let i = 0; i < n; i += step) {
    const tx = add("text", { x: x(i), y: H - 12, "text-anchor": "middle" }, axis);
    tx.textContent = i + 1;
  }
  const xl = add("text", { x: (W + m.l) / 2, y: H - 1, "text-anchor": "middle" }, axis);
  xl.textContent = "epoch";
  if (bestEpoch) {
    add("line", { x1: x(bestEpoch - 1), x2: x(bestEpoch - 1), y1: m.t, y2: H - m.b, stroke: "#74839b", "stroke-dasharray": "2 4" });
  }
  SERIES.forEach((s) => {
    const pts = data[s.key].map((v, i) => `${x(i)},${y(v)}`).join(" ");
    add("polyline", { points: pts, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", ...(s.dashed ? { "stroke-dasharray": "6 4" } : {}) });
    data[s.key].forEach((v, i) => add("circle", { cx: x(i), cy: y(v), r: n > 20 ? 0 : 3, fill: s.color, stroke: "#111827", "stroke-width": 2 }));
  });
  const cross = add("line", { y1: m.t, y2: H - m.b, stroke: "#a9b6cb", "stroke-width": 1, opacity: 0 });
  const hit = add("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });

  const chart = $("#loss-chart");
  chart.innerHTML = "";
  chart.append(svg);
  const tip = el("div", { class: "chart-tip hidden" });
  chart.append(tip);
  hit.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const px = ((ev.clientX - r.left) / r.width) * W;
    const i = Math.max(0, Math.min(n - 1, Math.round(((px - m.l) / (W - m.l - m.r)) * (n - 1))));
    cross.setAttribute("x1", x(i)); cross.setAttribute("x2", x(i)); cross.setAttribute("opacity", 0.6);
    tip.innerHTML = "";
    tip.append(el("div", { class: "mono", text: `epoch ${i + 1}${i + 1 === bestEpoch ? " · selected" : ""}` }));
    SERIES.forEach((s) => tip.append(el("div", { class: "row" },
      el("span", {}, el("i", { class: "dot", style: `background:${s.color}` }), s.label),
      el("span", { class: "mono", text: data[s.key][i].toFixed(4) }))));
    tip.classList.remove("hidden");
    const left = (x(i) / W) * r.width;
    tip.style.left = `${left > r.width / 2 ? left - 210 : left + 12}px`;
    tip.style.top = "10px";
  });
  hit.addEventListener("mouseleave", () => { tip.classList.add("hidden"); cross.setAttribute("opacity", 0); });

  const legend = $("#loss-legend");
  legend.innerHTML = "";
  SERIES.forEach((s) => legend.append(el("span", { class: "key" },
    el("span", { class: `line${s.dashed ? " dashed" : ""}`, style: s.dashed ? `border-color:${s.color}` : `background:${s.color}` }), s.label)));
  legend.append(el("span", { class: "key muted", text: "┆ selected epoch" }));

  const rows = data.train.map((_, i) => el("tr", {}, el("td", { class: "num", text: i + 1 }),
    ...SERIES.map((s) => el("td", { class: "num", text: data[s.key][i].toFixed(4) }))));
  const table = el("table", { class: "table" }, el("thead", {}, el("tr", {}, el("th", { class: "num", text: "epoch" }),
    ...SERIES.map((s) => el("th", { class: "num", text: s.label })))), el("tbody", {}, ...rows));
  $("#loss-table").innerHTML = "";
  $("#loss-table").append(el("div", { class: "table-scroll" }, table));
}
$("#chart-table-toggle").addEventListener("click", () => {
  const t = $("#loss-table");
  t.classList.toggle("hidden");
  $("#chart-table-toggle").textContent = t.classList.contains("hidden") ? "Show table" : "Hide table";
});

function renderValidationTables(t) {
  const c = t.curves;
  const exp = (v) => Math.exp(Math.min(v, 20));
  const rows = c.val_selection.map((agg, i) => el("tr", { style: i + 1 === t.training.best_epoch ? "background:rgba(213,81,129,.08)" : "" },
    el("td", { class: "num", text: `${i + 1}${i + 1 === t.training.best_epoch ? " ★" : ""}` }),
    ...["tinystories", "everyday", "kb"].map((s) => el("td", { class: "num", text: `${c.val_loss[s][i].toFixed(3)} / ${exp(c.val_loss[s][i]).toFixed(2)}` })),
    el("td", { class: "num", text: `${agg.toFixed(3)} / ${exp(agg).toFixed(2)}` })));
  $("#val-epochs").innerHTML = "";
  $("#val-epochs").append(el("div", { class: "muted small", text: "Each cell: validation loss / perplexity. ★ = saved epoch (lowest selection loss; effective ties keep the earlier epoch). KB is a diagnostic only." }),
    el("div", { class: "table-scroll" }, el("table", { class: "table" },
      el("thead", {}, el("tr", {}, el("th", { class: "num", text: "epoch" }), el("th", { class: "num", text: "TinyStories" }),
        el("th", { class: "num", text: "Everyday" }), el("th", { class: "num", text: "KB (diagnostic)" }), el("th", { class: "num", text: "selection" }))),
      el("tbody", {}, ...rows))));

  const v = t.validation.sources;
  const srcRows = Object.entries(v).map(([s, r]) => el("tr", {},
    el("td", { text: s === "kb" ? `${SOURCE_LABELS[s]} (diagnostic — not used for selection)` : SOURCE_LABELS[s] }),
    el("td", { class: "num", text: fmt(r.loss, 3) }), el("td", { class: "num", text: fmt(r.perplexity, 2) }),
    el("td", { class: "num", text: fmt(r.unigram_reference.loss, 3) }), el("td", { class: "num", text: fmt(r.unigram_reference.perplexity, 2) }),
    el("td", { class: "num", text: fmtInt(r.target_tokens) })));
  $("#val-sources").innerHTML = "";
  $("#val-sources").append(el("table", { class: "table" },
    el("thead", {}, el("tr", {}, el("th", { text: "validation source" }), el("th", { class: "num", text: "loss" }), el("th", { class: "num", text: "ppl" }),
      el("th", { class: "num", text: "unigram loss" }), el("th", { class: "num", text: "unigram ppl" }), el("th", { class: "num", text: "chars scored" }))),
    el("tbody", {}, ...srcRows)),
    el("div", { class: "muted small", text: "These are validation numbers: they were used to choose the saved epoch, so they are not an unbiased estimate. See the final held-out section below." }));

  const mix = t.training.mixture;
  const mixRows = Object.keys(mix.repeats).map((s) => el("tr", {},
    el("td", { text: SOURCE_LABELS[s] }), el("td", { class: "num", text: `×${mix.repeats[s]}` }),
    el("td", { class: "num", text: fmtInt(mix.target_tokens_unique[s]) }), el("td", { class: "num", text: `${(100 * mix.effective_share[s]).toFixed(1)}%` })));
  const man = t.data_report.manifest;
  const windows = man.report.windows;
  const winRows = ["tinystories", "everyday", "kb"].map((s) => {
    const w = windows[s].train;
    return el("tr", {}, el("td", { text: SOURCE_LABELS[s] }), el("td", { class: "num", text: fmtInt(w.examples) }),
      el("td", { class: "num", text: fmtInt(w.fit_directly) }), el("td", { class: "num", text: fmtInt(w.turn_windowed + w.chunked) }),
      el("td", { class: "num", text: fmtInt(w.windows) }), el("td", { class: "num", text: fmtInt(w.dropped) }));
  });
  $("#mixture").innerHTML = "";
  $("#mixture").append(
    el("table", { class: "table" }, el("thead", {}, el("tr", {}, el("th", { text: "source" }), el("th", { class: "num", text: "repeat" }),
      el("th", { class: "num", text: "unique target chars" }), el("th", { class: "num", text: "share / epoch" }))), el("tbody", {}, ...mixRows)),
    el("h3", { text: "Training windows (context " + man.context_length + ")" }),
    el("table", { class: "table" }, el("thead", {}, el("tr", {}, el("th", { text: "source" }), el("th", { class: "num", text: "examples" }),
      el("th", { class: "num", text: "fit" }), el("th", { class: "num", text: "windowed" }), el("th", { class: "num", text: "windows" }),
      el("th", { class: "num", text: "dropped" }))), el("tbody", {}, ...winRows)),
    el("div", { class: "muted small", text: `TinyStoriesInstruct: seeded, content-blind sample of ${fmtInt(man.tinystories_train_sample)} of ${fmtInt(man.report.tinystories.train_file_records)} records (public synthetic dataset, GPT-3.5/4 generated). Everyday Conversations: public synthetic dataset generated by Llama-3.1-70B. Knowledge base: hand-written for this project.` }));
}

function sampleCard(tag, s, extra = []) {
  return el("div", { class: "sample" },
    el("div", { class: "tag", text: tag }),
    s.history_turns ? el("div", { class: "lbl", text: `${s.history_turns} earlier dataset turns given as context` }) : null,
    el("div", { class: "lbl", text: "prompt" }), el("pre", { text: s.message }),
    el("div", { class: "lbl", text: "NanoLlama (generated)" }), el("pre", { text: s.reply == null ? "(prompt too long for context)" : s.reply || "(empty: immediate end-of-sequence)" }),
    ...extra);
}

function renderSamples(v) {
  const box = $("#val-samples");
  box.innerHTML = "";
  box.append(el("p", { class: "muted small", text: v.note }));
  v.samples.forEach((s) => box.append(sampleCard(s.set, s, [el("div", { class: "lbl", text: "dataset reference reply" }), el("pre", { text: s.reference })])));
  v.probes.forEach((p) => box.append(sampleCard(`probe · ${p.kind} · ${p.persona} · loop guard off`, p,
    p.reply_loop_guard_on != null ? [el("div", { class: "lbl", text: "diagnostic: same prompt with the loop guard on" }), el("pre", { text: p.reply_loop_guard_on })] : [])));
}

const FINAL_NOTES = {
  tinystories: "Held-out test: seeded subset of the official TinyStories validation file, disjoint from our validation subset.",
  everyday: "Held-out test: official Everyday Conversations test split.",
  kb: "Unseen KB entries (answers never seen in training). Loss at or above the unigram baseline means poor factual transfer.",
  kb_rephrase: "Answer seen during training; wording held out. The near-zero loss is teacher-forced on a memorized answer (each next character is predicted given the true previous ones); it shows memorization, not the ability to answer new questions.",
};

function renderFinal(t) {
  const badge = $("#final-badge");
  const box = $("#final-content");
  box.innerHTML = "";
  if (!t.finalized) {
    badge.textContent = "not finalized";
    badge.className = "badge pending";
    box.append(el("div", { class: "empty-state" },
      el("p", { text: `Version ${shortVersion(t.version)} has not been scored on the held-out test sets.` }),
      el("p", { html: "Run <code>python scripts/finalize.py</code> once, after choosing the production model. Retrained versions start unfinalized and never inherit an earlier version's test results." })));
    return;
  }
  const f = t.final;
  badge.textContent = `finalized${f.finalize_count > 1 ? ` · evaluated ${f.finalize_count}× (forced reruns recorded)` : ""}`;
  badge.className = "badge done";
  const rows = Object.entries(f.sources).map(([s, r]) => el("tr", {},
    el("td", {}, el("div", { text: SOURCE_LABELS[s] }), el("div", { class: "muted small", text: FINAL_NOTES[s] || r.label })),
    el("td", { class: "num", text: fmt(r.loss, 3) }), el("td", { class: "num", text: fmt(r.perplexity, 2) }),
    el("td", { class: "num", text: fmt(r.unigram_reference.loss, 3) }),
    el("td", { class: "num", text: r.mean_similarity == null ? "–" : r.mean_similarity.toFixed(3) }),
    el("td", { class: "num", text: fmtInt(r.target_tokens) })));
  box.append(el("p", { class: "lede", text: `${f.split}. Scored once by scripts/finalize.py for this version only.` }),
    el("table", { class: "table" }, el("thead", {}, el("tr", {}, el("th", { text: "held-out set" }), el("th", { class: "num", text: "loss" }),
      el("th", { class: "num", text: "ppl" }), el("th", { class: "num", text: "unigram loss" }),
      el("th", { class: "num", "data-tip": f.similarity_definition, text: "mean similarity ?" }), el("th", { class: "num", text: "chars scored" }))),
      el("tbody", {}, ...rows)),
    el("p", { class: "muted small", text: `Similarity: ${f.similarity_definition} It measures character overlap with the reference, not semantic correctness: a reply can score well while being wrong, or poorly while being acceptable.` }),
    el("p", { class: "muted small", text: "Reading: TinyStories and Everyday test losses sit far below the context-free unigram baseline, so the model learned story and small-talk character patterns. On unseen KB entries it does not beat the baseline: it has not acquired transferable facts. The rephrasing diagnostic reflects memorized training answers. None of these numbers indicate a useful general assistant." }));
  const det = el("details", { class: "samples" }, el("summary", { text: `Held-out generation samples (${f.decoding_for_samples})` }));
  Object.entries(f.sources).forEach(([s, r]) => r.samples.forEach((smp) => det.append(sampleCard(`${SOURCE_LABELS[s]}${smp.similarity != null ? ` · similarity ${smp.similarity.toFixed(2)}` : ""}`, smp,
    [el("div", { class: "lbl", text: "reference" }), el("pre", { text: smp.reference })]))));
  box.append(det);
}

/* ------------------------------------------------------------------ architecture (S05) */
const GLOSSARY = [
  ["Token", "The unit the model reads and writes. Here one token is one character, plus six special control tokens."],
  ["Embedding", "A learned vector for each token id. NanoLlama reuses the same matrix as its output layer (weight tying)."],
  ["Logits", "Raw scores for every possible next token; softmax turns them into probabilities."],
  ["Cross-entropy loss", "Average of −log(probability the model gave to the correct next character), in nats. Lower is better."],
  ["Perplexity", "exp(loss): roughly how many characters the model is choosing between at each step."],
  ["SFT (as used here)", "Supervised instruction training from random initialization: the loss is computed only on assistant replies. It is not fine-tuning of a pretrained model."],
  ["Loss masking", "Prompt, system and padding positions are excluded from the loss, so the model learns to produce replies, not to predict the user's text."],
  ["Causal mask", "Stops each position from attending to later positions, so the model can only use the past when predicting the next character."],
  ["Attention head", "One independent set of query/key/value projections; each head can learn a different pattern of what to look at."],
  ["Validation set", "Examples held out from training and used to pick the saved epoch. Because it guides that choice, it is not a final estimate."],
  ["Final held-out test", "Examples never used for training or selection, scored once by an explicit finalize step."],
  ["Unigram baseline", "A model that predicts each character from its overall frequency without looking at context."],
  ["Greedy decoding", "Always pick the most likely next character (used when temperature ≤ 0.15)."],
  ["Synthetic dataset", "Text generated by a larger language model rather than written by people."],
];

async function loadArchitecture() {
  const flow = $("#arch-flow");
  const gl = $("#glossary");
  gl.innerHTML = "";
  GLOSSARY.forEach(([k, v]) => gl.append(el("dt", { text: k }), el("dd", { text: v })));
  let a;
  try {
    a = await api("/api/architecture");
  } catch (err) {
    flow.innerHTML = `<li><div class="banner banner-error">Could not load architecture: ${err.message}</div></li>`;
    return;
  }
  if (!a.available) {
    $("#arch-lede").textContent = `No model loaded (${a.reason}). The diagram below needs a promoted checkpoint.`;
    flow.innerHTML = "";
    return;
  }
  const c = a.config;
  $("#arch-lede").textContent = `Dimensions read from the served checkpoint (${shortVersion(a.version)}): ${fmtInt(a.parameters)} trainable parameters. B = batch, T = sequence length ≤ ${c.context_length}.`;
  const stages = [
    ["Token embedding + rotary angles", `Character ids → ${c.d_model}-dim vectors from a ${c.vocab_size} × ${c.d_model} table. cos/sin tables for RoPE are precomputed for ${c.context_length} positions × ${a.head_dim} dims (base ${c.rope_base}).`, `[B, T] → [B, T, ${c.d_model}]`],
    ["RMSNorm → causal self-attention", `${c.n_heads} heads × ${a.head_dim} dims. Bias-free Q, K, V, O projections; RoPE on Q and K; upper-triangular mask; KV cache while generating.`, `[B, T, ${c.d_model}] → [B, ${c.n_heads}, T, ${a.head_dim}] → [B, T, ${c.d_model}]`],
    ["Residual add", "The attention output is added back to its input.", `x = x + Attn(RMSNorm(x))`],
    ["RMSNorm → SwiGLU feed-forward", `W₁, W₃: ${c.d_model} → ${c.d_ff}; W₂: ${c.d_ff} → ${c.d_model}; SiLU gate.`, `[B, T, ${c.d_model}] → [B, T, ${c.d_ff}] → [B, T, ${c.d_model}]`],
    ["Residual add", `Stages 2–5 form one block, repeated ${c.n_layers} times.`, `x = x + FFN(RMSNorm(x))`],
    ["Final RMSNorm + tied LM head", `Projection back to ${c.vocab_size} logits using the embedding matrix (weight tying).`, `[B, T, ${c.d_model}] → [B, T, ${c.vocab_size}]`],
  ];
  flow.innerHTML = "";
  stages.forEach(([t, d, s], i) => flow.append(el("li", { class: i >= 1 && i <= 4 ? "repeat" : "" },
    el("div", { class: "t", text: t }), el("div", { class: "d", text: d }), el("div", { class: "s", text: s }))));
}

/* ------------------------------------------------------------------ retraining (F10) */
const rt = { epochs: $("#rt-epochs"), batch: $("#rt-batch"), lr: $("#rt-lr"), budget: $("#rt-budget") };
[["epochs", (v) => v], ["batch", (v) => v], ["lr", (v) => Number(v).toFixed(4)]].forEach(([k, f]) => {
  const upd = () => { $(`#rt-${k}-out`).textContent = f(rt[k].value); };
  rt[k].addEventListener("input", upd);
  upd();
});
let pollTimer = null;
function setRetrainControls(disabled) {
  Object.values(rt).forEach((i) => { i.disabled = disabled; });
  $("#rt-start").disabled = disabled;
}
function openRetrain() {
  $("#retrain-modal").classList.remove("hidden");
  pollRetrain();
}
function closeRetrain() {
  $("#retrain-modal").classList.add("hidden");
}
$("#open-retrain").addEventListener("click", openRetrain);
$("#close-retrain").addEventListener("click", closeRetrain);

function renderProgress(job) {
  const box = $("#rt-progress");
  if (job.status === "idle") { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.innerHTML = "";
  job.epochs.forEach((e) => box.append(el("div", {
    text: `epoch ${e.epoch}/${e.epochs}  train ${e.train_loss.toFixed(3)}  selection ${e.val_selection.toFixed(3)}  kb(diag) ${e.val_loss.kb.toFixed(3)}${e.best ? " ★" : ""}  ${e.seconds.toFixed(0)}s`,
  })));
  if (job.status === "running") {
    const next = job.epochs.length + 1;
    const total = job.params.epochs;
    box.append(el("div", { class: "running", text: next <= total ? `optimizing weights: epoch ${next}/${total} … (${Math.round(job.elapsed_seconds)}s elapsed)` : "writing, verifying and promoting the new version …" }));
  }
}

async function pollRetrain() {
  clearTimeout(pollTimer);
  let job;
  try { job = await api("/api/retrain"); } catch (err) { return; }
  renderProgress(job);
  if (job.status === "running") {
    setRetrainControls(true);
    pollTimer = setTimeout(pollRetrain, 2000);
  } else {
    setRetrainControls(false);
  }
  return job;
}

let lastSeenJobFinish = null;
$("#rt-start").addEventListener("click", async () => {
  $("#rt-error").classList.add("hidden");
  $("#rt-success").classList.add("hidden");
  setRetrainControls(true);
  try {
    await api("/api/retrain", { method: "POST", body: JSON.stringify({
      epochs: parseInt(rt.epochs.value, 10), batch_size: parseInt(rt.batch.value, 10), lr: parseFloat(rt.lr.value), budget: rt.budget.value,
    }) });
  } catch (err) {
    $("#rt-error").textContent = err.message;
    $("#rt-error").classList.remove("hidden");
    setRetrainControls(false);
    return;
  }
  const wait = async () => {
    const job = await pollRetrain();
    if (!job || job.status === "running") { setTimeout(wait, 2000); return; }
    if (job.finished_at === lastSeenJobFinish) return;
    lastSeenJobFinish = job.finished_at;
    if (job.status === "succeeded") {
      $("#rt-success").textContent = `New version ${shortVersion(job.version)} promoted and loaded. It is not finalized.`;
      $("#rt-success").classList.remove("hidden");
      confetti();
      await refreshHealth();
      Object.keys(loaded).forEach((k) => { if (k !== "chat") loaded[k] = false; });
      const active = $(".tab.active").dataset.tab;
      if (active !== "chat") { loaded[active] = true; ({ attention: loadAttention, tokenizer: loadTokenizer, telemetry: loadTelemetry, architecture: loadArchitecture }[active])(); }
      setTimeout(closeRetrain, 3500);
    } else if (job.status === "failed") {
      $("#rt-error").textContent = `Training failed; the previous model is still being served. ${job.error}`;
      $("#rt-error").classList.remove("hidden");
    }
  };
  setTimeout(wait, 1500);
});

function confetti() {
  const canvas = $("#confetti");
  const ctx = canvas.getContext("2d");
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const colors = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f472b6"];
  const parts = Array.from({ length: 160 }, () => ({
    x: canvas.width / 2 + (Math.random() - 0.5) * 200, y: canvas.height / 3,
    vx: (Math.random() - 0.5) * 12, vy: Math.random() * -12 - 2, s: 4 + Math.random() * 5,
    c: colors[Math.floor(Math.random() * colors.length)], r: Math.random() * 6,
  }));
  let frame = 0;
  const tick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    parts.forEach((p) => {
      p.vy += 0.35; p.x += p.vx; p.y += p.vy; p.r += 0.1;
      ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.r); ctx.fillStyle = p.c; ctx.fillRect(-p.s / 2, -p.s / 2, p.s, p.s * 0.6); ctx.restore();
    });
    if (++frame < 150) requestAnimationFrame(tick); else ctx.clearRect(0, 0, canvas.width, canvas.height);
  };
  tick();
}

/* ------------------------------------------------------------------ boot */
resetChat();
loadDecodingDefaults();
refreshHealth();
loadPresets();
pollRetrain();
