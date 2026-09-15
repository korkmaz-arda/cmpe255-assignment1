"""Deterministic quick-capture tokenizer.

One free-text line becomes a title plus structured attributes:

    !urgent   priority (first ``!word``)
    #tag      tags (every occurrence, lowercased, de-duplicated)
    ~1.5h     estimate (first ``~number[unit]``; h/hr multiply by 60)
    @friday   due date (first ``@token``)

Recognized-kind tokens are removed from the title even when their value is
not understood (``@someday`` yields no date but still disappears). Tags are
scanned on the original input; the other extractors run on progressively
stripped text. No network calls, no model inference.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

PRIORITY_SYNONYMS = {
    "urgent": "urgent", "crit": "urgent", "critical": "urgent", "u": "urgent",
    "high": "high", "h": "high", "important": "high",
    "medium": "medium", "med": "medium", "m": "medium", "normal": "medium",
    "low": "low", "l": "low",
}

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# A token must start at the beginning of the line or after whitespace, so
# e-mail addresses ("a@b.com") or "C#" are not treated as tokens.
TAG_RE = re.compile(r"(?<!\S)#([A-Za-z0-9_-]+)")
PRIORITY_RE = re.compile(r"(?<!\S)!([A-Za-z0-9_]+)")
ESTIMATE_RE = re.compile(r"(?<!\S)~(\d+(?:\.\d+)?|\.\d+)(hrs|hr|h|mins|min|m)?(?![\w.])", re.IGNORECASE)
# Trailing sentence punctuation ("@friday," / "@tomorrow.") is not part of the value.
DUE_RE = re.compile(r"(?<!\S)@(\S+?)(?=[.,;:!?)\]]*(?:\s|$))")
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass
class CaptureResult:
    title: str
    priority: str = "medium"
    tags: list[str] = field(default_factory=list)
    estimated_minutes: int | None = None
    due_date: date | None = None
    chips: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "title": self.title,
            "priority": self.priority,
            "tags": list(self.tags),
            "estimated_minutes": self.estimated_minutes,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "chips": list(self.chips),
        }


def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))


def resolve_due_token(value: str, today: date) -> date | None:
    v = value.lower()
    if v in ("today", "tod"):
        return today
    if v in ("tomorrow", "tom"):
        return today + timedelta(days=1)
    if v == "yesterday":
        return today - timedelta(days=1)
    if v == "nextweek":
        return today + timedelta(days=7)
    if ISO_RE.match(v):
        try:
            return date.fromisoformat(v)
        except ValueError:
            return None
    for index, name in enumerate(WEEKDAYS):
        if v == name or v == name[:3]:
            # Next occurrence strictly after today.
            delta = (index - today.weekday()) % 7 or 7
            return today + timedelta(days=delta)
    return None


def _strip(text: str, match: re.Match) -> str:
    return text[: match.start()] + " " + text[match.end():]


def parse_capture(text: str, today: date) -> CaptureResult:
    original = text or ""
    chips: list[dict] = []

    # Tags: scanned against the original input.
    tags: list[str] = []
    for m in TAG_RE.finditer(original):
        name = m.group(1).lower()
        if name not in tags:
            tags.append(name)
    for name in tags:
        chips.append({"kind": "tag", "label": f"#{name}", "value": name, "recognized": True})
    working = TAG_RE.sub(" ", original)

    priority = "medium"
    m = PRIORITY_RE.search(working)
    if m:
        word = m.group(1).lower()
        if word in PRIORITY_SYNONYMS:
            priority = PRIORITY_SYNONYMS[word]
            chips.append({"kind": "priority", "label": priority.capitalize(), "value": priority, "recognized": True})
        else:
            chips.append({"kind": "priority", "label": f"!{m.group(1)} (unknown)", "value": None, "recognized": False})
        working = _strip(working, m)

    estimate = None
    m = ESTIMATE_RE.search(working)
    if m:
        amount = float(m.group(1))
        unit = (m.group(2) or "").lower()
        estimate = _round_half_up(amount * 60 if unit in ("h", "hr", "hrs") else amount)
        chips.append({"kind": "estimate", "label": f"{estimate} min", "value": estimate, "recognized": True})
        working = _strip(working, m)

    due = None
    m = DUE_RE.search(working)
    if m:
        due = resolve_due_token(m.group(1), today)
        if due is not None:
            chips.append({"kind": "due", "label": due.isoformat(), "value": due.isoformat(), "recognized": True})
        else:
            chips.append({"kind": "due", "label": f"@{m.group(1)} (unknown)", "value": None, "recognized": False})
        working = _strip(working, m)

    title = " ".join(working.split())
    return CaptureResult(title=title, priority=priority, tags=tags,
                         estimated_minutes=estimate, due_date=due, chips=chips)
