import asyncio

import pytest
from starlette.testclient import TestClient

from app.events import EventHub, format_sse
from app.server import create_app


@pytest.fixture
def app(db_path, clock):
    return create_app(db_path, clock=clock, seed=False)


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def events(app, monkeypatch):
    published = []
    original = app.state.hub.publish

    def record(event_type, **data):
        published.append(event_type)
        original(event_type, **data)

    monkeypatch.setattr(app.state.hub, "publish", record)
    return published


def test_index_and_static_served(client):
    assert "Zenith" in client.get("/").text
    assert client.get("/static/js/app.js").status_code == 200


def test_seeded_app_starts_with_demo_content(db_path, clock):
    ws = TestClient(create_app(db_path, clock=clock)).get("/api/workspace").json()
    assert len(ws["tasks"]) == 5 and ws["today"] == "2026-09-16"


def test_capture_crud_and_events(client, events):
    r = client.post("/api/tasks", json={"capture": "Deploy !high #ops ~1h @tomorrow"})
    assert r.status_code == 201
    task = r.json()
    assert (task["title"], task["priority"], task["due_date"], task["bucket"]) == ("Deploy", "high", "2026-09-17", "tomorrow")
    tid = task["id"]

    assert client.patch(f"/api/tasks/{tid}", json={"status": "completed"}).json()["completed_at"] == "2026-09-16T10:00:00"
    full = client.put(f"/api/tasks/{tid}", json={
        "title": "Deploy v2", "description": "", "status": "todo", "priority": "low", "category_id": None,
        "due_date": None, "estimated_minutes": None, "time_spent_minutes": 0, "pinned": False,
        "subtasks": [{"title": "step", "completed": False}], "tag_ids": []}).json()
    assert full["completed_at"] is None and full["subtasks_total"] == 1 and full["tags"] == []
    assert client.post(f"/api/tasks/{tid}/matrix", json={"quadrant": "do_first"}).json()["quadrant"] == "do_first"
    assert client.post(f"/api/tasks/{tid}/focus", json={"minutes": 25}).json()["time_spent_minutes"] == 25
    assert client.post("/api/tasks/reorder", json={"items": [{"id": tid, "order_index": 1}], "status": "review"}).json()[0]["status"] == "review"
    assert client.post(f"/api/tasks/{tid}/archive").json()["archived"] is True
    assert client.post(f"/api/tasks/{tid}/unarchive").json()["archived"] is False
    assert client.delete(f"/api/tasks/{tid}").json()["outcome"] == "trashed"
    assert client.post(f"/api/tasks/{tid}/restore").json()["deleted"] is False
    assert client.post("/api/tasks/bulk", json={"action": "complete", "ids": [tid]}).json() == {"count": 1}
    assert client.delete(f"/api/tasks/{tid}?permanent=1").json()["outcome"] == "purged"
    assert client.get(f"/api/tasks/{tid}").status_code == 404

    assert events == ["task_created", "task_updated", "task_updated", "task_updated", "task_updated",
                      "tasks_reordered", "task_updated", "task_updated", "task_trashed", "task_updated",
                      "batch_completed", "task_deleted"]


def test_validation_errors_are_400(client):
    assert client.post("/api/tasks", json={"capture": "!high"}).status_code == 400
    assert client.post("/api/tasks", content="not json", headers={"content-type": "application/json"}).status_code == 400
    assert client.get("/api/workspace?scope=nope").status_code == 400
    assert client.get("/api/workspace?sort=nope").status_code == 400
    assert client.post("/api/tasks/bulk", json={"action": "complete", "ids": []}).status_code == 400
    assert client.delete("/api/categories/999").status_code == 404


def test_workspace_filters_and_counts(client):
    cat = client.post("/api/categories", json={"name": "Work"}).json()
    client.post("/api/tasks", json={"capture": "Report today @today", "category_id": cat["id"]})
    client.post("/api/tasks", json={"capture": "Later @nextweek #x"})
    trashed = client.post("/api/tasks", json={"capture": "Old"}).json()
    client.delete(f"/api/tasks/{trashed['id']}")

    ws = client.get("/api/workspace", params={"scope": "trash"}).json()
    assert [t["title"] for t in ws["tasks"]] == ["Old"]
    # counts describe every destination even while viewing Trash
    assert ws["counts"]["smart"]["all"] == 2 and ws["counts"]["trash"] == 1
    assert ws["counts"]["categories"] == {str(cat["id"]): 1}

    ws = client.get("/api/workspace", params={"category": cat["id"], "q": "REPORT"}).json()
    assert [t["title"] for t in ws["tasks"]] == ["Report today"]


def test_tags_and_categories_endpoints(client, events):
    first = client.post("/api/tags", json={"name": "#Infra", "color": "#6366f1"})
    assert first.status_code == 201 and first.json()["name"] == "infra"
    dup = client.post("/api/tags", json={"name": "infra"})
    assert dup.status_code == 200 and dup.json()["created"] is False and dup.json()["id"] == first.json()["id"]
    task = client.post("/api/tasks", json={"capture": "Tagged #infra"}).json()
    assert client.delete(f"/api/tags/{first.json()['id']}").status_code == 204
    assert client.get(f"/api/tasks/{task['id']}").json()["tags"] == []
    cat = client.post("/api/categories", json={"name": "Tmp"}).json()
    assert cat["icon"] == "folder" and cat["color"].startswith("#")
    assert "tags_updated" in events and "categories_updated" in events


def test_analytics_and_calendar_and_activity(client):
    client.post("/api/tasks", json={"capture": "One"})
    a = client.get("/api/analytics").json()
    assert a["total"] == 1 and len(a["heatmap"]) == 30
    cal = client.get("/api/calendar", params={"year": 2026, "month": 2}).json()
    assert cal["label"] == "February 2026"
    assert client.get("/api/calendar").json()["month"] == 9
    assert client.get("/api/activity?limit=1").json()[0]["action"] == "created"
    assert client.delete("/api/activity").json()["cleared"] >= 1
    assert client.get("/api/activity").json() == []


def test_parse_preview_endpoint(client):
    parsed = client.post("/api/parse", json={"text": "Plan !u #a @fri ~2h"}).json()
    assert parsed == {**parsed, "title": "Plan", "priority": "urgent", "tags": ["a"], "due_date": "2026-09-18",
                      "estimated_minutes": 120}


def test_calendar_default_due_via_api(client):
    t = client.post("/api/tasks", json={"capture": "Dentist", "default_due": "2026-10-02"}).json()
    assert t["due_date"] == "2026-10-02"


def test_event_hub_fanout():
    async def scenario():
        hub = EventHub()
        a, b = hub.subscribe(), hub.subscribe()
        hub.publish("task_created", id=1)
        assert (await a.get())["type"] == "task_created"
        assert (await b.get()) == {"type": "task_created", "id": 1}
        hub.unsubscribe(a)
        assert hub.subscriber_count == 1
        with pytest.raises(ValueError):
            hub.publish("nonsense")
        small = EventHub(queue_size=2)
        q = small.subscribe()
        for i in range(5):
            small.publish("task_updated", id=i)
        assert q.qsize() >= 1  # overflow never raises
    asyncio.run(scenario())
    assert format_sse({"type": "x"}) == 'data: {"type": "x"}\n\n'


def test_event_hub_close_ends_streams():
    async def scenario():
        hub = EventHub()
        q = hub.subscribe()
        hub.publish("task_created", id=1)
        hub.close()
        assert await q.get() is None  # pending events dropped; sentinel ends the stream
        assert await hub.subscribe().get() is None  # late subscribers end immediately
    asyncio.run(scenario())
