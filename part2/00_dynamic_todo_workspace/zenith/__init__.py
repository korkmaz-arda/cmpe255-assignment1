"""Zenith task workspace core.

Framework-independent domain logic: storage, quick-capture parsing,
derived classifications (buckets, Eisenhower quadrants), filtering and
sorting, and productivity telemetry. Nothing here imports the web layer.
"""

PRIORITIES = ("urgent", "high", "medium", "low")
STATUSES = ("todo", "in_progress", "review", "completed")
PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}


class ValidationError(ValueError):
    """Raised when submitted task data violates a domain rule."""


class NotFoundError(LookupError):
    """Raised when a referenced task, category, or tag does not exist."""
