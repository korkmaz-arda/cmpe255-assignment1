// Synthesized audio cues (Web Audio) and a canvas confetti burst.
// Audio is created lazily inside user gestures to respect autoplay rules;
// any failure degrades silently. Each cue that plays dispatches a
// "zenith:feedback" event so the behavior is observable in tests.

import { safeStorage, saveStorage } from "./dom.js";

let ctx = null;
let muted = safeStorage("zenith.muted", "0") === "1";

export const isMuted = () => muted;

export function setMuted(value) {
  muted = value;
  saveStorage("zenith.muted", value ? "1" : "0");
}

function audio() {
  if (ctx) return ctx;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) return null;
  try {
    ctx = new Ctor();
  } catch {
    ctx = null;
  }
  return ctx;
}

// Call from a user gesture so later cues are allowed to play.
export function unlockAudio() {
  const c = audio();
  if (c && c.state === "suspended") c.resume().catch(() => {});
}

function tone(c, { freq, endFreq, start, duration, type = "sine", gain = 0.12, filter }) {
  const osc = c.createOscillator();
  const amp = c.createGain();
  osc.type = type;
  osc.frequency.setValueAtTime(freq, start);
  if (endFreq) osc.frequency.exponentialRampToValueAtTime(endFreq, start + duration);
  amp.gain.setValueAtTime(0.0001, start);
  amp.gain.exponentialRampToValueAtTime(gain, start + 0.015);
  amp.gain.exponentialRampToValueAtTime(0.0001, start + duration);
  let node = osc;
  if (filter) {
    const f = c.createBiquadFilter();
    f.type = "lowpass";
    f.frequency.setValueAtTime(filter, start);
    f.frequency.exponentialRampToValueAtTime(200, start + duration);
    osc.connect(f);
    node = f;
  }
  node.connect(amp).connect(c.destination);
  osc.start(start);
  osc.stop(start + duration + 0.02);
}

const CUES = {
  complete: (c, t) => [523.25, 659.25, 783.99, 1046.5].forEach((f, i) =>
    tone(c, { freq: f, start: t + i * 0.08, duration: 0.25, type: "triangle" })),
  uncomplete: (c, t) => tone(c, { freq: 440, endFreq: 220, start: t, duration: 0.18, type: "sine" }),
  click: (c, t) => tone(c, { freq: 900, start: t, duration: 0.05, type: "square", gain: 0.04 }),
  fanfare: (c, t) => [392, 523.25, 659.25, 783.99].forEach((f, i) =>
    tone(c, { freq: f, start: t + i * 0.14, duration: i === 3 ? 0.6 : 0.2, type: "sawtooth", gain: 0.07, filter: 3000 })),
  delete: (c, t) => tone(c, { freq: 600, endFreq: 90, start: t, duration: 0.35, type: "sawtooth", gain: 0.08, filter: 1800 }),
};

export function play(kind) {
  if (muted || !CUES[kind]) return;
  document.dispatchEvent(new CustomEvent("zenith:feedback", { detail: { kind } }));
  try {
    const c = audio();
    if (!c) return;
    if (c.state === "suspended") c.resume().catch(() => {});
    CUES[kind](c, c.currentTime + 0.01);
  } catch {
    /* audio unavailable: silent */
  }
}

export function confetti() {
  if (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const canvas = document.createElement("canvas");
  canvas.className = "confetti-canvas";
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  document.body.append(canvas);
  const g = canvas.getContext("2d");
  if (!g) {
    canvas.remove();
    return;
  }
  const colors = ["#6366f1", "#06b6d4", "#10b981", "#f59e0b", "#f43f5e", "#ec4899"];
  const parts = Array.from({ length: 140 }, () => ({
    x: canvas.width / 2 + (Math.random() - 0.5) * 120,
    y: canvas.height * 0.45,
    vx: (Math.random() - 0.5) * 16,
    vy: -Math.random() * 16 - 4,
    size: Math.random() * 6 + 4,
    rot: Math.random() * Math.PI,
    spin: (Math.random() - 0.5) * 0.3,
    color: colors[Math.floor(Math.random() * colors.length)],
  }));
  const started = performance.now();
  function frame(now) {
    const elapsed = now - started;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const p of parts) {
      p.vy += 0.45;
      p.vx *= 0.99;
      p.x += p.vx;
      p.y += p.vy;
      p.rot += p.spin;
      g.save();
      g.globalAlpha = Math.max(0, 1 - elapsed / 1800);
      g.translate(p.x, p.y);
      g.rotate(p.rot);
      g.fillStyle = p.color;
      g.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
      g.restore();
    }
    if (elapsed < 1800) requestAnimationFrame(frame);
    else canvas.remove();
  }
  requestAnimationFrame(frame);
}
