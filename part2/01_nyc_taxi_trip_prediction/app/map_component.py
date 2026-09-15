"""Estimator route map (F09) as a bidirectional Streamlit component.

Primary view: a Leaflet street map (standard OpenStreetMap tiles) where a click sets the
endpoint chosen by the explicit pickup/dropoff mode, and both markers can be dragged. Landmark shortcuts
stay on the map. The map needs internet access for the Leaflet library (jsDelivr) and the tiles (tile.openstreetmap.org).

If the Leaflet library cannot be loaded, the component falls back to the offline stylised SVG map (landmark
clicks only). If only the tiles fail, the street map still returns exact coordinates on a blank background.

The component only reports raw lat/lon events. Python validates them and computes every feature.
"""
from __future__ import annotations

import streamlit as st

from taxi import landmarks

LEAFLET_VERSION = "1.9.4"
# Standard OpenStreetMap tiles: keyless, light interactive use only (OSMF tile usage policy), attribution required.
TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_ATTRIBUTION = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'


# Offline fallback frame (lat/lon) and canvas; outlines are hand-simplified, not survey-accurate.
FRAME = {"lat_min": 40.60, "lat_max": 40.86, "lon_min": -74.09, "lon_max": -73.73}
WIDTH, HEIGHT = 600, 570
BOROUGHS = {
    "Manhattan": [(40.700, -74.019), (40.710, -73.978), (40.745, -73.968), (40.772, -73.943), (40.800, -73.928),
                  (40.835, -73.934), (40.872, -73.910), (40.879, -73.927), (40.850, -73.947), (40.800, -73.972),
                  (40.760, -74.005), (40.720, -74.015)],
    "Bronx": [(40.800, -73.928), (40.785, -73.915), (40.810, -73.870), (40.805, -73.790), (40.870, -73.785),
              (40.915, -73.830), (40.905, -73.910), (40.879, -73.927), (40.872, -73.910), (40.835, -73.934)],
    "Queens": [(40.738, -73.955), (40.760, -73.935), (40.785, -73.915), (40.800, -73.840), (40.790, -73.770),
               (40.760, -73.705), (40.650, -73.725), (40.600, -73.740), (40.590, -73.800), (40.620, -73.820),
               (40.640, -73.860), (40.680, -73.870), (40.700, -73.900), (40.730, -73.925)],
    "Brooklyn": [(40.700, -73.995), (40.690, -74.030), (40.640, -74.040), (40.575, -74.010), (40.570, -73.940),
                 (40.585, -73.860), (40.640, -73.860), (40.680, -73.870), (40.700, -73.900), (40.730, -73.925),
                 (40.738, -73.955), (40.710, -73.975)],
    "Staten Island": [(40.640, -74.075), (40.648, -74.180), (40.560, -74.195), (40.505, -74.195),
                      (40.540, -74.130), (40.590, -74.060)],
}
BOROUGH_LABELS = {"Manhattan": (40.820, -73.975), "Bronx": (40.845, -73.865), "Queens": (40.715, -73.800),
                  "Brooklyn": (40.650, -73.960), "Staten Island": (40.612, -74.035)}
AIRPORT_AREAS = [("JFK", 40.6413, -73.7781, 0.020), ("LGA", 40.7769, -73.8740, 0.012)]
# Label side per landmark so dense Midtown labels do not collide.
LABEL_SIDE = {"times_square": "left", "empire_state": "left", "grand_central": "right", "central_park": "left",
              "wall_street": "left", "brooklyn_bridge": "right", "williamsburg": "right", "astoria": "right",
              "lga": "right", "jfk": "left"}

_JS = r"""
const LEAFLET_JS = "https://cdn.jsdelivr.net/npm/leaflet@__LV__/dist/leaflet-src.esm.js";
const LEAFLET_CSS = "https://cdn.jsdelivr.net/npm/leaflet@__LV__/dist/leaflet.css";
const LOAD_TIMEOUT_MS = 8000;
const seq = () => `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

function loadLeaflet() {
  if (!window.__taxiLeafletPromise) {
    if (!document.querySelector("link[data-taxi-leaflet]")) {
      const link = document.createElement("link");
      link.rel = "stylesheet"; link.href = LEAFLET_CSS; link.setAttribute("data-taxi-leaflet", "1");
      document.head.appendChild(link);
    }
    window.__taxiLeafletPromise = import(LEAFLET_JS);
  }
  return window.__taxiLeafletPromise;
}

function status(host, text, kind) {
  const s = host.querySelector(".taxi-status");
  if (!text) { s.style.display = "none"; s.textContent = ""; s.dataset.kind = ""; return; }
  s.style.display = "block"; s.textContent = text; s.dataset.kind = kind || "info";
}

function emit(host, payload) { host.__latest.setTriggerValue("map_event", { ...payload, seq: seq() }); }

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let host = parentElement.querySelector(".taxi-map-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "taxi-map-host";
    host.innerHTML = '<div class="taxi-status" style="display:none"></div>' +
                     '<div class="taxi-leaflet" style="display:none"></div><div class="taxi-svg"></div>';
    parentElement.appendChild(host);
  }
  host.__latest = { data, setTriggerValue };
  if (host.__mode === "leaflet") { updateLeaflet(host); return; }
  if (host.__mode === "svg") { renderSvg(host); return; }
  if (host.__loading) return;
  host.__loading = true;
  status(host, "Loading street map…", "info");
  const timeout = new Promise((_, reject) => setTimeout(() => reject(new Error("timeout")), LOAD_TIMEOUT_MS));
  Promise.race([loadLeaflet(), timeout])
    .then((L) => { host.__L = L; host.__mode = "leaflet"; status(host, ""); initLeaflet(host); updateLeaflet(host); })
    .catch(() => {
      window.__taxiLeafletPromise = null;
      host.__mode = "svg";
      renderSvg(host);
    });
}

// ---------------------------------------------------------------- Leaflet street map
function pinIcon(L, which) {
  const letter = which === "pickup" ? "P" : "D";
  return L.divIcon({ className: "taxi-pin-wrap", iconSize: [30, 30], iconAnchor: [15, 15],
    html: `<div class="taxi-pin ${which}"><span class="pulse"></span><span class="core">${letter}</span></div>` });
}

function initLeaflet(host) {
  const L = host.__L, d = host.__latest.data;
  const box = host.querySelector(".taxi-leaflet");
  box.style.display = "block";
  host.querySelector(".taxi-svg").innerHTML = "";
  const map = L.map(box, { zoomControl: true, attributionControl: true, minZoom: 9, maxZoom: 18 });
  map.setView([40.73, -73.93], 11);
  const tiles = L.tileLayer(d.tile_url, { maxZoom: 19, attribution: d.tile_attribution });
  let loaded = 0, failed = 0;
  tiles.on("tileload", () => { loaded++; if (host.dataset.tileWarning) { delete host.dataset.tileWarning; status(host, ""); } });
  tiles.on("tileerror", () => {
    failed++;
    if (loaded === 0) {
      host.dataset.tileWarning = "1";
      status(host, "Street tiles could not be loaded (no internet?). Clicks still set exact coordinates; " +
                   "landmark shortcuts and the coordinate fields keep working.", "warn");
    }
  });
  tiles.addTo(map);
  map.on("click", (e) => emit(host, { type: "click", lat: e.latlng.lat, lng: e.latlng.lng }));

  const lmLayer = L.layerGroup().addTo(map);
  const lmMarkers = {};
  for (const lm of d.landmarks) {
    const side = d.label_side[lm.id] || "right";
    const m = L.circleMarker([lm.lat, lm.lon], { radius: 6, weight: 2, color: "#0d0d0d", fillColor: "#c3c2b7",
      fillOpacity: 1, bubblingMouseEvents: false, className: "taxi-lm" })
      .bindTooltip(`${lm.icon} ${lm.name}`, { permanent: true, direction: side, offset: [side === "left" ? -8 : 8, 0],
        className: "taxi-lm-label" })
      .on("click", () => emit(host, { type: "landmark", id: lm.id }))
      .addTo(lmLayer);
    lmMarkers[lm.id] = m;
  }

  const glow = L.polyline([], { color: "#ffffff", weight: 9, opacity: 0.7, interactive: false }).addTo(map);
  const route = L.polyline([], { color: "#184f95", weight: 4, dashArray: "8 6", interactive: false }).addTo(map);
  const markers = {};
  for (const which of ["pickup", "dropoff"]) {
    markers[which] = L.marker([0, 0], { icon: pinIcon(L, which), draggable: true, autoPan: true,
      zIndexOffset: which === "pickup" ? 1000 : 900, bubblingMouseEvents: false })
      .bindTooltip(which === "pickup" ? "Pickup" : "Dropoff", { direction: "top", offset: [0, -16],
        className: `taxi-pin-label ${which}` })
      .on("dragend", (ev) => { const ll = ev.target.getLatLng(); emit(host, { type: "drag", endpoint: which, lat: ll.lat, lng: ll.lng }); })
      .addTo(map);
  }
  const taxi = L.marker([0, 0], { interactive: false, keyboard: false, zIndexOffset: 800,
    icon: L.divIcon({ className: "taxi-car-wrap", iconSize: [22, 12], iconAnchor: [11, 6],
                      html: '<div class="taxi-car"><span></span></div>' }) }).addTo(map);

  const ModeControl = L.Control.extend({ onAdd() { const div = L.DomUtil.create("div", "taxi-mode"); return div; } });
  const modeCtl = new ModeControl({ position: "topright" }).addTo(map);
  const noteCtl = new (L.Control.extend({ onAdd() {
    const div = L.DomUtil.create("div", "taxi-note");
    div.textContent = "Dashed line = straight-line connection, not the driven route";
    return div; } }))({ position: "bottomleft" }).addTo(map);

  new ResizeObserver(() => map.invalidateSize()).observe(box);
  host.__map = { map, tiles, markers, route, glow, taxi, lmMarkers, modeCtl, lastKey: null };
  host.__tileStats = () => ({ loaded, failed });

  let t0 = performance.now();
  const step = (now) => {
    const s = host.__map, dd = host.__latest.data;
    if (!document.body.contains(box)) return;  // unmounted
    const p = L.latLng(dd.pickup.lat, dd.pickup.lon), q = L.latLng(dd.dropoff.lat, dd.dropoff.lon);
    const period = dd.anim_seconds * 1000;
    const f = ((now - t0) % period) / period;
    const pos = L.latLng(p.lat + (q.lat - p.lat) * f, p.lng + (q.lng - p.lng) * f);
    s.taxi.setLatLng(pos);
    const a = map.latLngToLayerPoint(p), b = map.latLngToLayerPoint(q);
    const el = s.taxi.getElement() && s.taxi.getElement().querySelector(".taxi-car");
    if (el) {
      el.style.transform = `rotate(${Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI}deg)`;
      el.style.display = p.equals(q) ? "none" : "block";
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function updateLeaflet(host) {
  const L = host.__L, d = host.__latest.data, s = host.__map;
  if (!s) return;
  const p = L.latLng(d.pickup.lat, d.pickup.lon), q = L.latLng(d.dropoff.lat, d.dropoff.lon);
  s.markers.pickup.setLatLng(p).setTooltipContent(`Pickup · ${d.pickup.label}`);
  s.markers.dropoff.setLatLng(q).setTooltipContent(`Dropoff · ${d.dropoff.label}`);
  s.route.setLatLngs([p, q]); s.glow.setLatLngs([p, q]);
  const ctl = s.modeCtl.getContainer();
  ctl.className = `taxi-mode leaflet-control ${d.mode}`;
  ctl.innerHTML = `Map click sets: <b>${d.mode === "pickup" ? "PICKUP" : "DROPOFF"}</b>`;
  const box = host.querySelector(".taxi-leaflet");
  box.classList.toggle("mode-pickup", d.mode === "pickup");
  box.classList.toggle("mode-dropoff", d.mode === "dropoff");
  for (const [id, m] of Object.entries(s.lmMarkers)) {
    const role = id === d.pickup.choice ? "#0ca30c" : id === d.dropoff.choice ? "#d03b3b" : "#c3c2b7";
    m.setStyle({ fillColor: role });
  }
  const key = `${p.lat},${p.lng}|${q.lat},${q.lng}`;
  if (s.lastKey === null) {
    s.map.fitBounds(L.latLngBounds([p, q]).pad(0.3), { maxZoom: 14 });
  } else if (key !== s.lastKey) {
    const view = s.map.getBounds();
    if (!view.contains(p) || !view.contains(q)) s.map.fitBounds(L.latLngBounds([p, q]).pad(0.3), { maxZoom: 14 });
  }
  s.lastKey = key;
  host.dataset.pickup = `${p.lat},${p.lng}`;
  host.dataset.dropoff = `${q.lat},${q.lng}`;
  host.dataset.mode = d.mode;
}

// ---------------------------------------------------------------- offline SVG fallback
function renderSvg(host) {
  const d = host.__latest.data, NS = "http://www.w3.org/2000/svg";
  const F = d.frame, W = d.width, H = d.height, PAD = 14;
  host.querySelector(".taxi-leaflet").style.display = "none";
  status(host, "Street map unavailable: the map library could not be loaded (it needs internet). Showing the " +
               "offline stylised map. Click a landmark to set the " + d.mode.toUpperCase() +
               ", or type exact coordinates.", "warn");
  host.dataset.pickup = `${d.pickup.lat},${d.pickup.lon}`;
  host.dataset.dropoff = `${d.dropoff.lat},${d.dropoff.lon}`;
  host.dataset.mode = d.mode;
  const proj = (lat, lon) => {
    const x = (lon - F.lon_min) / (F.lon_max - F.lon_min) * W, y = (F.lat_max - lat) / (F.lat_max - F.lat_min) * H;
    return [Math.min(W - PAD, Math.max(PAD, x)), Math.min(H - PAD, Math.max(PAD, y))];
  };
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    if (parent) parent.appendChild(e);
    return e;
  };
  const root = host.querySelector(".taxi-svg");
  root.innerHTML = "";
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Offline stylised route map" });
  root.appendChild(svg);
  el("rect", { x: 0, y: 0, width: W, height: H, class: "water" }, svg);
  for (const [name, pts] of Object.entries(d.boroughs)) {
    el("polygon", { points: pts.map(([a, b]) => proj(a, b).join(",")).join(" "), class: "land" }, svg);
    const [lx, ly] = proj(...d.borough_labels[name]);
    el("text", { x: lx, y: ly, class: "borough" }, svg).textContent = name.toUpperCase();
  }
  for (const [label, lat, lon, rdeg] of d.airports) {
    const [cx, cy] = proj(lat, lon), r = rdeg / (F.lat_max - F.lat_min) * H;
    el("circle", { cx, cy, r, class: "airport" }, svg);
    el("text", { x: cx, y: cy - r - 4, class: "airport-label" }, svg).textContent = label;
  }
  const [px, py] = proj(d.pickup.lat, d.pickup.lon), [dx, dy] = proj(d.dropoff.lat, d.dropoff.lon);
  const path = `M ${px} ${py} L ${dx} ${dy}`;
  if (Math.hypot(dx - px, dy - py) > 1) {
    el("path", { d: path, class: "route-glow" }, svg);
    el("path", { d: path, class: "route" }, svg);
  }
  for (const lm of d.landmarks) {
    const [x, y] = proj(lm.lat, lm.lon);
    const g = el("g", { class: "node", "data-id": lm.id, tabindex: 0 }, svg);
    el("circle", { cx: x, cy: y, r: 14, class: "hit" }, g);
    el("circle", { cx: x, cy: y, r: 5, class: "dot" }, g);
    const left = d.label_side[lm.id] === "left";
    el("text", { x: x + (left ? -10 : 10), y: y + 4, "text-anchor": left ? "end" : "start", class: "node-label" }, g).textContent = lm.name;
    el("title", {}, g).textContent = `${lm.name} (${lm.zone}) — click to set as ${d.mode}`;
    const pick = () => emit(host, { type: "landmark", id: lm.id });
    g.addEventListener("click", pick);
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") pick(); });
  }
  for (const [which, x, y] of [["pickup", px, py], ["dropoff", dx, dy]]) {
    el("circle", { cx: x, cy: y, r: 9, class: `end ${which}` }, svg);
    el("text", { x, y: y + 4, class: "end-letter" }, svg).textContent = which === "pickup" ? "P" : "D";
  }
}
""".replace("__LV__", LEAFLET_VERSION)

_CSS = """
.taxi-map-host { position: relative; }
.taxi-status { margin: 0 0 8px; padding: 8px 12px; border-radius: 8px; font: 13px system-ui, sans-serif;
  border: 1px solid rgba(255,255,255,0.15); color: #c3c2b7; background: #1a1a19; }
.taxi-status[data-kind="warn"] { border-color: #fab219; color: #fab219; }
.taxi-leaflet { height: 580px; width: 100%; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10);
  background: #0f1b2a; }
.taxi-leaflet.mode-pickup { cursor: crosshair; outline: 2px solid rgba(12,163,12,0.55); }
.taxi-leaflet.mode-dropoff { cursor: crosshair; outline: 2px solid rgba(208,59,59,0.60); }
.taxi-leaflet.leaflet-container { cursor: crosshair; }
.taxi-leaflet .leaflet-tile-pane { filter: brightness(0.82) saturate(0.85); }  /* soften light tiles in the dark UI */
.taxi-leaflet .leaflet-control-attribution { background: rgba(255,255,255,0.85); }
.taxi-mode { padding: 6px 10px; border-radius: 8px; font: 13px system-ui, sans-serif; color: #fff;
  background: rgba(13,13,13,0.85); border: 2px solid #898781; }
.taxi-mode.pickup { border-color: #0ca30c; } .taxi-mode.pickup b { color: #7fd67f; }
.taxi-mode.dropoff { border-color: #d03b3b; } .taxi-mode.dropoff b { color: #f08a8a; }
.taxi-note { padding: 3px 8px; border-radius: 6px; font: 11px system-ui, sans-serif; color: #c3c2b7;
  background: rgba(13,13,13,0.75); }
.taxi-lm-label { background: rgba(13,13,13,0.80); color: #e8e7e1; border: 0; box-shadow: none;
  font: 11px system-ui, sans-serif; padding: 1px 5px; }
.taxi-lm-label::before { display: none; }
.taxi-pin-label { font: 600 12px system-ui, sans-serif; }
.taxi-pin { position: relative; width: 30px; height: 30px; }
.taxi-pin .core { position: absolute; inset: 5px; border-radius: 50%; display: flex; align-items: center;
  justify-content: center; font: 700 11px system-ui, sans-serif; color: #fff; border: 2px solid #0d0d0d;
  cursor: grab; }
.taxi-pin.pickup .core { background: #0ca30c; } .taxi-pin.dropoff .core { background: #d03b3b; }
.taxi-pin .pulse { position: absolute; inset: 0; border-radius: 50%; animation: taxi-pulse 1.8s ease-out infinite; }
.taxi-pin.pickup .pulse { border: 2px solid #0ca30c; } .taxi-pin.dropoff .pulse { border: 2px solid #d03b3b; }
@keyframes taxi-pulse { 0% { transform: scale(0.6); opacity: 0.9; } 100% { transform: scale(1.5); opacity: 0; } }
.taxi-car { width: 22px; height: 12px; border-radius: 3px; background: #f7c600; border: 1px solid #0b0b0b;
  position: relative; }
.taxi-car span { position: absolute; left: 8px; top: 2px; width: 6px; height: 6px; background: #0b0b0b; border-radius: 1px; }
.taxi-svg svg { width: 100%; height: auto; display: block; border-radius: 12px; border: 1px solid rgba(255,255,255,0.10); }
.taxi-svg .water { fill: #0f1b2a; } .taxi-svg .land { fill: #1d2633; stroke: #2f3b4b; stroke-width: 1.2; }
.taxi-svg .borough { fill: #5d6b7d; font: 600 10px system-ui, sans-serif; letter-spacing: 0.12em; text-anchor: middle; }
.taxi-svg .airport { fill: rgba(57,135,229,0.12); stroke: #3987e5; stroke-dasharray: 4 3; }
.taxi-svg .airport-label { fill: #9ec5f4; font: 600 10px system-ui, sans-serif; text-anchor: middle; }
.taxi-svg .route-glow { fill: none; stroke: #fab219; stroke-opacity: 0.3; stroke-width: 9; }
.taxi-svg .route { fill: none; stroke: #fab219; stroke-width: 3; stroke-dasharray: 8 6; }
.taxi-svg .node { cursor: pointer; } .taxi-svg .node .hit { fill: transparent; }
.taxi-svg .node .dot { fill: #c3c2b7; stroke: #0f1b2a; stroke-width: 2; }
.taxi-svg .node-label { fill: #c3c2b7; font: 10.5px system-ui, sans-serif; paint-order: stroke; stroke: #0f1b2a; stroke-width: 3px; }
.taxi-svg .end.pickup { fill: #0ca30c; } .taxi-svg .end.dropoff { fill: #d03b3b; }
.taxi-svg .end { stroke: #0d0d0d; stroke-width: 2; pointer-events: none; }
.taxi-svg .end-letter { fill: #fff; font: 700 10px system-ui, sans-serif; text-anchor: middle; pointer-events: none; }
"""

_route_map = st.components.v2.component("nyc_route_map", css=_CSS, js=_JS, isolate_styles=False)


def route_map(pickup: dict, dropoff: dict, mode: str, duration_seconds: float | None, key: str, on_event) -> None:
    """Mount the map. `pickup`/`dropoff`: {lat, lon, label, choice}. Events arrive as the `map_event` trigger."""
    # Taxi animation length loosely follows the predicted duration (compressed to 3-10 s for display).
    anim = 4.0 if duration_seconds is None else max(3.0, min(10.0, duration_seconds / 180.0))
    payload = {
        "mode": mode,
        "pickup": pickup, "dropoff": dropoff,
        "anim_seconds": round(anim, 2),
        "tile_url": TILE_URL, "tile_attribution": TILE_ATTRIBUTION,
        "landmarks": [{"id": lm.id, "name": lm.name, "lat": lm.lat, "lon": lm.lon, "zone": lm.zone, "icon": lm.icon}
                      for lm in landmarks.LANDMARKS],
        "label_side": LABEL_SIDE,
        "frame": FRAME, "width": WIDTH, "height": HEIGHT,
        "boroughs": {k: [list(p) for p in v] for k, v in BOROUGHS.items()},
        "borough_labels": {k: list(v) for k, v in BOROUGH_LABELS.items()},
        "airports": [list(a) for a in AIRPORT_AREAS],
    }
    _route_map(data=payload, key=key, on_map_event_change=on_event)
