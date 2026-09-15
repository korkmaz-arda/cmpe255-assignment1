"""Injectable date/time source.

The server is the only authority for "today" (overdue status, weekday
resolution, Eisenhower urgency, analytics windows). Production uses the
system's local time; tests inject a fixed clock.
"""

from __future__ import annotations

from datetime import date, datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now().replace(microsecond=0)

    def today(self) -> date:
        return self.now().date()


class FixedClock:
    """Clock pinned to a given moment; ``advance`` supports multi-day scenarios."""

    def __init__(self, moment: datetime | str):
        if isinstance(moment, str):
            moment = datetime.fromisoformat(moment)
        self._now = moment.replace(microsecond=0)

    def now(self) -> datetime:
        return self._now

    def today(self) -> date:
        return self._now.date()

    def set(self, moment: datetime | str) -> None:
        if isinstance(moment, str):
            moment = datetime.fromisoformat(moment)
        self._now = moment.replace(microsecond=0)


def stamp(moment: datetime) -> str:
    """Timestamps are stored as local ISO-8601 strings (``YYYY-MM-DDTHH:MM:SS``)."""
    return moment.strftime("%Y-%m-%dT%H:%M:%S")
