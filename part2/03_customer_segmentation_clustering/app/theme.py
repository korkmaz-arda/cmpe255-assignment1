"""Shared presentation helpers: CSS, cards and small formatting utilities."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1400px; }

.app-header {
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding:1rem 1.3rem; border-radius:14px; margin-bottom:1rem;
  background:linear-gradient(120deg,#1e1b4b 0%,#312e81 55%,#4c1d95 100%); color:#fff;
}
.app-header h1 { font-size:1.35rem; margin:0; font-weight:700; letter-spacing:-.01em; }
.app-header .sub { font-size:.82rem; opacity:.82; margin-top:.15rem; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
  background:#34d399; margin-right:.4rem; vertical-align:middle; }
.status-dot.off { background:#f87171; }

.panel { border:1px solid rgba(148,163,184,.28); border-radius:12px;
  padding:1rem 1.15rem; margin-bottom:.9rem; background:rgba(148,163,184,.05); }
.panel h3 { margin:0 0 .5rem 0; font-size:1rem; }

.persona-card { border:1px solid rgba(148,163,184,.3); border-left-width:5px;
  border-radius:11px; padding:.75rem .85rem; height:100%; }
.persona-card .idx { font-size:.7rem; text-transform:uppercase; letter-spacing:.07em; opacity:.65; }
.persona-card .nm { font-weight:700; font-size:1.02rem; margin:.15rem 0 .1rem; }
.persona-card .tl { font-size:.78rem; opacity:.75; min-height:2.3em; line-height:1.25; }
.persona-card .stat { display:flex; justify-content:space-between; font-size:.79rem;
  padding:.16rem 0; border-top:1px dashed rgba(148,163,184,.25); }
.persona-card .stat b { font-variant-numeric:tabular-nums; }

.chip { display:inline-block; padding:.2rem .6rem; border-radius:999px; font-size:.72rem;
  border:1px solid rgba(148,163,184,.35); margin:.12rem .25rem .12rem 0; }
.chip code { font-size:.72rem; background:transparent; }

.badge { display:inline-block; padding:.12rem .5rem; border-radius:6px;
  font-size:.7rem; font-weight:700; letter-spacing:.02em; }
.badge.prod { background:rgba(16,185,129,.18); color:#10b981; border:1px solid rgba(16,185,129,.4); }
.badge.lead { background:rgba(124,58,237,.18); color:#a78bfa; border:1px solid rgba(124,58,237,.4); }
.badge.plain { background:rgba(148,163,184,.14); color:#94a3b8; border:1px solid rgba(148,163,184,.3); }
.badge.fail { background:rgba(239,68,68,.15); color:#f87171; border:1px solid rgba(239,68,68,.35); }
.badge.acc { background:rgba(16,185,129,.18); color:#10b981; }
.badge.rej { background:rgba(239,68,68,.16); color:#f87171; }

.result-card { border-radius:12px; padding:1rem 1.15rem; color:#fff; }
.result-card .cap { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; opacity:.85; }
.result-card .nm { font-size:1.45rem; font-weight:800; margin:.1rem 0 .25rem; }
.result-card .desc { font-size:.83rem; opacity:.93; line-height:1.4; }
.callout { background:rgba(255,255,255,.16); border-radius:9px; padding:.6rem .75rem;
  margin-top:.7rem; font-size:.83rem; line-height:1.4; }

.caveat { border-left:3px solid #f59e0b; background:rgba(245,158,11,.09);
  padding:.65rem .9rem; border-radius:0 9px 9px 0; font-size:.82rem; line-height:1.45; }
.note { font-size:.78rem; opacity:.72; line-height:1.4; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.78rem; }

div[data-testid="stMetricValue"] { font-size:1.5rem; }
</style>
"""


def compact(markup: str) -> str:
    """Collapse an HTML block to a single line before handing it to markdown.

    Streamlit renders HTML through a markdown parser, and a markdown parser
    treats a blank line as a paragraph break: it closes the HTML block there and
    renders whatever follows — typically a trailing ``</div>`` — as literal text
    on the page. Conditional f-string interpolations are the usual culprit,
    because an unmet condition leaves a whitespace-only line behind.

    Joining the non-empty lines with a single space removes that failure mode
    entirely while preserving word spacing in any prose inside the block.
    """
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def panel(title: str, body_html: str) -> None:
    st.markdown(compact(f"<div class='panel'><h3>{title}</h3>{body_html}</div>"),
                unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render a block of HTML safely through the compactor."""
    st.markdown(compact(markup), unsafe_allow_html=True)


def money(value: float) -> str:
    return f"${value:,.0f}"


def metric_or_dash(value, fmt: str = "{:.4f}") -> str:
    return "—" if value is None else fmt.format(value)
