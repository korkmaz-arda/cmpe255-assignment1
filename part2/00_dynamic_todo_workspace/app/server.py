"""Thin Starlette HTTP + SSE layer over the ``zenith`` core.

Routes validate transport-level shape only; all domain rules live in the
core. Every mutation publishes a typed event so open clients refetch.
Routes are async and run the (fast, local) SQLite work on the event-loop
thread, which also serializes writes within the process.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from zenith import NotFoundError, ValidationError, analytics, views
from zenith.capture import parse_capture
from zenith.clock import SystemClock
from zenith.db import Database
from zenith.store import CATEGORY_COLORS, CATEGORY_ICONS, TAG_COLORS, Store

from .events import EventHub, format_sse

STATIC_DIR = Path(__file__).parent / "static"
KEEPALIVE_SECONDS = 15


def _int_param(value: str | None, name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        raise ValidationError(f"{name} must be an integer") from None


async def _body(request: Request) -> dict:
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValidationError("Request body must be JSON") from None
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object")
    return data


def create_app(db_path: str | Path, *, clock=None, seed: bool = True) -> Starlette:
    store = Store(Database(db_path), clock or SystemClock())
    if seed:
        store.seed_if_first_run()
    else:
        store.mark_seeded()
    hub = EventHub()

    def task_id_of(request: Request) -> int:
        return int(request.path_params["task_id"])

    # ----------------------------------------------------------------- reads
    async def index(request: Request):
        return FileResponse(STATIC_DIR / "index.html")

    async def workspace(request: Request):
        q = request.query_params
        today = store.today()
        tasks = store.all_tasks()
        working = views.filter_tasks(
            tasks, today, scope=q.get("scope", "active"), smart=q.get("smart", "all"),
            category_id=_int_param(q.get("category"), "category"), tag_id=_int_param(q.get("tag"), "tag"),
            search=q.get("q", ""))
        working = views.sort_tasks(working, q.get("sort", "custom"))
        return JSONResponse({
            "today": today.isoformat(),
            "tasks": [views.decorate(t, today) for t in working],
            "counts": views.sidebar_counts(tasks, today),
            "categories": store.list_categories(),
            "tags": store.list_tags(),
            "palettes": {"category": CATEGORY_COLORS, "tag": TAG_COLORS, "icons": CATEGORY_ICONS},
        })

    async def analytics_view(request: Request):
        return JSONResponse(analytics.compute(store.all_tasks(), store.list_categories(), store.today()))

    async def calendar_view(request: Request):
        today = store.today()
        year = _int_param(request.query_params.get("year"), "year") or today.year
        month = _int_param(request.query_params.get("month"), "month") or today.month
        return JSONResponse(views.month_grid(year, month, today))

    async def parse(request: Request):
        data = await _body(request)
        text = data.get("text", "")
        if not isinstance(text, str):
            raise ValidationError("text must be a string")
        return JSONResponse(parse_capture(text, store.today()).as_dict())

    async def get_task(request: Request):
        return JSONResponse(store.get_task(task_id_of(request)))

    async def activity(request: Request):
        limit = _int_param(request.query_params.get("limit"), "limit") or 50
        return JSONResponse(store.activity(limit))

    # ------------------------------------------------------------- mutations
    async def create_task(request: Request):
        data = await _body(request)
        if "capture" in data:
            category_id = data.get("category_id")
            task = store.create_from_capture(
                data["capture"], category_id=category_id if category_id not in ("", None) else None,
                status=data.get("status"), default_due=data.get("default_due") or None)
        else:
            task = store.create_task(data)
        hub.publish("task_created", id=task["id"])
        return JSONResponse(task, status_code=201)

    async def update_task(request: Request):
        task = store.update_task(task_id_of(request), await _body(request))
        hub.publish("task_updated", id=task["id"])
        return JSONResponse(task)

    async def patch_task(request: Request):
        task = store.patch_task(task_id_of(request), await _body(request))
        hub.publish("task_updated", id=task["id"])
        return JSONResponse(task)

    async def delete_task(request: Request):
        task_id = task_id_of(request)
        outcome = store.delete_task(task_id, permanent=request.query_params.get("permanent") in ("1", "true"))
        hub.publish("task_deleted" if outcome == "purged" else "task_trashed", id=task_id)
        return JSONResponse({"id": task_id, "outcome": outcome})

    def lifecycle(method):
        async def handler(request: Request):
            task = method(task_id_of(request))
            hub.publish("task_updated", id=task["id"])
            return JSONResponse(task)
        return handler

    async def matrix_drop(request: Request):
        data = await _body(request)
        task = store.move_to_quadrant(task_id_of(request), data.get("quadrant"))
        hub.publish("task_updated", id=task["id"])
        return JSONResponse(task)

    async def focus(request: Request):
        data = await _body(request)
        task = store.add_focus_minutes(task_id_of(request), data.get("minutes"))
        hub.publish("task_updated", id=task["id"])
        return JSONResponse(task)

    async def reorder(request: Request):
        data = await _body(request)
        tasks = store.reorder(data.get("items"), data.get("status"))
        hub.publish("tasks_reordered", ids=[t["id"] for t in tasks])
        return JSONResponse(tasks)

    async def bulk(request: Request):
        data = await _body(request)
        action = data.get("action")
        count = store.bulk(action, data.get("ids"), data.get("value"))
        hub.publish("batch_completed" if action == "complete" else "batch_updated", action=action, count=count)
        return JSONResponse({"count": count})

    async def create_category(request: Request):
        data = await _body(request)
        category = store.create_category(data.get("name"), data.get("color"), data.get("icon") or "folder")
        hub.publish("categories_updated")
        return JSONResponse(category, status_code=201)

    async def delete_category(request: Request):
        store.delete_category(int(request.path_params["category_id"]))
        hub.publish("categories_updated")
        return Response(status_code=204)

    async def create_tag(request: Request):
        data = await _body(request)
        tag, created = store.create_tag(data.get("name"), data.get("color"))
        if created:
            hub.publish("tags_updated")
        return JSONResponse({**tag, "created": created}, status_code=201 if created else 200)

    async def delete_tag(request: Request):
        store.delete_tag(int(request.path_params["tag_id"]))
        hub.publish("tags_updated")
        return Response(status_code=204)

    async def clear_activity(request: Request):
        count = store.clear_activity()
        hub.publish("activity_cleared")
        return JSONResponse({"cleared": count})

    # ------------------------------------------------------------------- SSE
    async def events(request: Request):
        queue = hub.subscribe()

        async def stream():
            try:
                yield "retry: 2000\n" + format_sse({"type": "connected", "today": store.today().isoformat()})
                while True:
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
                        if message is None:  # server shutting down
                            break
                        yield format_sse(message)
                    except asyncio.TimeoutError:
                        if await request.is_disconnected():
                            break
                        yield ": keepalive\n\n"
            finally:
                hub.unsubscribe(queue)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def validation_error(request: Request, exc: ValidationError):
        return JSONResponse({"error": str(exc)}, status_code=400)

    async def not_found(request: Request, exc: NotFoundError):
        return JSONResponse({"error": str(exc)}, status_code=404)

    routes = [
        Route("/", index),
        Route("/api/workspace", workspace),
        Route("/api/analytics", analytics_view),
        Route("/api/calendar", calendar_view),
        Route("/api/parse", parse, methods=["POST"]),
        Route("/api/events", events),
        Route("/api/tasks", create_task, methods=["POST"]),
        Route("/api/tasks/reorder", reorder, methods=["POST"]),
        Route("/api/tasks/bulk", bulk, methods=["POST"]),
        Route("/api/tasks/{task_id:int}", get_task, methods=["GET"]),
        Route("/api/tasks/{task_id:int}", update_task, methods=["PUT"]),
        Route("/api/tasks/{task_id:int}", patch_task, methods=["PATCH"]),
        Route("/api/tasks/{task_id:int}", delete_task, methods=["DELETE"]),
        Route("/api/tasks/{task_id:int}/restore", lifecycle(store.restore_task), methods=["POST"]),
        Route("/api/tasks/{task_id:int}/archive", lifecycle(store.archive_task), methods=["POST"]),
        Route("/api/tasks/{task_id:int}/unarchive", lifecycle(store.unarchive_task), methods=["POST"]),
        Route("/api/tasks/{task_id:int}/matrix", matrix_drop, methods=["POST"]),
        Route("/api/tasks/{task_id:int}/focus", focus, methods=["POST"]),
        Route("/api/categories", create_category, methods=["POST"]),
        Route("/api/categories/{category_id:int}", delete_category, methods=["DELETE"]),
        Route("/api/tags", create_tag, methods=["POST"]),
        Route("/api/tags/{tag_id:int}", delete_tag, methods=["DELETE"]),
        Route("/api/activity", activity, methods=["GET"]),
        Route("/api/activity", clear_activity, methods=["DELETE"]),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    app = Starlette(routes=routes, exception_handlers={ValidationError: validation_error,
                                                       NotFoundError: not_found})
    app.state.store = store
    app.state.hub = hub
    return app

