"""In-process broadcast hub for Server-Sent Events.

Clients never merge event payloads: every event simply tells connected
browsers to refetch their working set from the server.
"""

from __future__ import annotations

import asyncio
import json

EVENT_TYPES = (
    "task_created", "task_updated", "task_trashed", "task_deleted", "tasks_reordered",
    "batch_completed", "batch_updated", "categories_updated", "tags_updated", "activity_cleared",
)


class EventHub:
    def __init__(self, queue_size: int = 100):
        self._subscribers: set[asyncio.Queue] = set()
        self._queue_size = queue_size
        self._loop: asyncio.AbstractEventLoop | None = None
        self.closed = False

    def subscribe(self) -> asyncio.Queue:
        self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.add(queue)
        if self.closed:
            queue.put_nowait(None)
        return queue

    def close(self) -> None:
        """End every open stream (a ``None`` sentinel) so the server can shut down promptly."""
        self.closed = True
        for queue in list(self._subscribers):
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(None)

    def close_threadsafe(self) -> None:
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self.close)

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event_type: str, **data) -> None:
        """Must be called from the event-loop thread (all routes are async)."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {event_type}")
        message = {"type": event_type, **data}
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                # A stalled client only needs to know "something changed"; drop the backlog.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait(message)


def format_sse(message: dict) -> str:
    return f"data: {json.dumps(message)}\n\n"
