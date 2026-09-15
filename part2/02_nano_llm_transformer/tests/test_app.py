import json
import shutil
import time

import pytest
from fastapi.testclient import TestClient

from app.engine import Engine
from app.server import create_app
from nanollama import artifacts
from nanollama.generate import DecodeSettings, fit_prompt, generate


@pytest.fixture()
def client(trained, prepared, tmp_path):
    arts, _ = trained
    root = tmp_path / "arts"
    shutil.copytree(arts, root)
    engine = Engine(artifacts_root=root, data_root=prepared, device="cpu")
    return TestClient(create_app(engine)), engine, root


def _stream(client, **body):
    res = client.post("/api/chat", json=body)
    assert res.status_code == 200, res.text
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


def test_health_and_index(client):
    c, engine, _ = client
    h = c.get("/api/health").json()
    assert h["model_loaded"] and h["vocab_size"] == 104 and h["parameters"] == engine.model.num_parameters()
    assert h["finalized"] is False
    assert "NanoLlama" in c.get("/").text


def test_chat_stream_is_model_generation(client):
    c, engine, _ = client
    body = dict(message="Hello there", system="You are NanoLlama, a small helpful assistant.", temperature=0.9,
                top_p=0.95, top_k=50, repetition_penalty=1.2, max_new_tokens=40, loop_guard=True, seed=11)
    events = _stream(c, **body)
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    tokens = [e for e in events if e["type"] == "token"]
    assert all({"id", "text", "count", "tokens_per_second", "ttft_ms"} <= set(e) for e in tokens)
    assert events[-1]["kv_cache"] is True
    # Wiring proof: the stream equals direct autoregressive generation with the same settings.
    s = DecodeSettings(temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.2, max_new_tokens=40, loop_guard=True, seed=11)
    ids, _ = fit_prompt(engine.tok, body["system"], body["message"], [], engine.model.cfg.context_length, 40)
    expected, _ = generate(engine.model, engine.tok, ids, s)
    assert [e["id"] for e in tokens] == expected
    # A control change yields a different reply.
    other = _stream(c, **{**body, "temperature": 0.0})
    assert [e["id"] for e in other if e["type"] == "token"] != [e["id"] for e in tokens]


def test_chat_validation_errors(client):
    c, _, _ = client
    assert c.post("/api/chat", json={"message": "hi", "temperature": 3.0}).status_code == 422
    assert c.post("/api/chat", json={"message": "x" * 300, "max_new_tokens": 200}).status_code == 422  # tiny context


def test_tokenize_and_attention_contracts(client):
    c, engine, _ = client
    t = c.post("/api/tokenize", json={"text": "Hi\nyou"}).json()
    assert t["total_tokens"] == 6 and t["characters"] == 6 and t["chars_per_token"] == 1.0
    assert [x["text"] for x in t["tokens"]] == list("Hi\nyou")
    assert len(t["top_predictions"]) == 5 and 0 < sum(p["percent"] for p in t["top_predictions"]) <= 100.0001
    assert len(c.get("/api/control-tokens").json()) == 6
    a = c.post("/api/attention", json={"text": "x" * 100}).json()
    cfg = engine.model.cfg
    assert a["n_layers"] == cfg.n_layers and a["n_heads"] == cfg.n_heads and a["size"] == 64 and a["truncated"]
    assert len(a["attention"]) == cfg.n_layers and len(a["attention"][0]) == cfg.n_heads
    assert len(a["attention"][0][0]) == 64 and a["tokens"][0] == "<|bos|>"


def test_telemetry_separates_validation_and_final(client, prepared):
    c, engine, root = client
    t = c.get("/api/telemetry").json()
    assert t["available"] and t["finalized"] is False and t["final"] is None
    assert "validation" in t["validation"]["split"]
    assert set(t["selected_validation"]["per_source_loss"]) == {"tinystories", "everyday", "kb"}
    from nanollama.finalize import finalize
    finalize(root, prepared, device="cpu", log=lambda *_: None)
    t2 = c.get("/api/telemetry").json()
    assert t2["finalized"] and "held-out" in t2["final"]["split"]
    arch = c.get("/api/architecture").json()
    assert arch["config"]["n_layers"] == engine.model.cfg.n_layers


def test_retrain_validation_and_hot_reload(client):
    c, engine, root = client
    assert c.post("/api/retrain", json={"epochs": 0, "batch_size": 8, "lr": 0.002}).status_code == 422
    assert c.post("/api/retrain", json={"epochs": 1, "batch_size": 128, "lr": 0.002}).status_code == 422
    assert c.post("/api/retrain", json={"epochs": 1, "batch_size": 8, "lr": 0.5}).status_code == 422
    old = c.get("/api/health").json()["version"]

    # Retrain with a tiny model config so it runs quickly on CPU.
    from nanollama import train as train_mod
    from tests.conftest import tiny_model_config
    real_cfg = train_mod.TrainConfig

    def tiny_cfg(**kw):
        return real_cfg(**{**kw, "model": tiny_model_config()})

    import app.engine as engine_mod
    engine_mod.TrainConfig = tiny_cfg
    try:
        res = c.post("/api/retrain", json={"epochs": 1, "batch_size": 8, "lr": 0.002, "budget": "quick"})
        assert res.status_code == 200 and res.json()["status"] == "running"
        for _ in range(600):
            job = c.get("/api/retrain").json()
            if job["status"] != "running":
                break
            time.sleep(0.1)
    finally:
        engine_mod.TrainConfig = real_cfg
    assert job["status"] == "succeeded", job["error"]
    assert len(job["epochs"]) == 1
    h = c.get("/api/health").json()
    assert h["version"] == job["version"] != old and h["finalized"] is False
    assert artifacts.current_version(root).name == job["version"]


def test_lazy_recovery_without_model(tmp_path, prepared, trained):
    root = tmp_path / "empty"
    root.mkdir()
    engine = Engine(artifacts_root=root, data_root=prepared, device="cpu")
    c = TestClient(create_app(engine))
    assert c.get("/api/health").json()["model_loaded"] is False
    assert c.post("/api/tokenize", json={"text": "hi"}).status_code == 503
    assert c.get("/api/telemetry").json()["available"] is False
    # A version appears later (e.g. after training in another process): no restart needed.
    arts, _ = trained
    shutil.copytree(arts / "versions", root / "versions")
    shutil.copy(arts / artifacts.POINTER, root / artifacts.POINTER)
    assert c.get("/api/health").json()["model_loaded"] is True
    assert c.post("/api/tokenize", json={"text": "hi"}).status_code == 200


def test_ui_api_default_decoding_matches_probe_and_final_eval_defaults(client, prepared):
    import re

    import nanollama.compare as cmp
    from app.server import ChatRequest, STATIC
    from nanollama.generate import canonical_greedy

    c, engine, root = client
    canonical = canonical_greedy().to_dict()
    assert canonical["temperature"] == 0.0 and canonical["repetition_penalty"] == 1.0 and canonical["loop_guard"] is False
    # API request defaults, served UI defaults, probe defaults all equal the canonical configuration
    req = ChatRequest(message="x").model_dump()
    assert {k: req[k] for k in canonical} == canonical
    assert c.get("/api/decoding-defaults").json()["default"] == canonical
    assert cmp.GREEDY.to_dict() == canonical
    # the static HTML controls mirror the same values before JS loads them
    html = (STATIC / "index.html").read_text()
    for field in ("temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"):
        value = re.search(rf'id="{field}"[^>]*value="([^"]+)"', html).group(1)
        assert float(value) == float(canonical[field]), field
    assert 'id="loop_guard" checked' not in html
    # final-evaluation samples use the same configuration
    from nanollama.finalize import finalize
    out = finalize(root, prepared, device="cpu", log=lambda *_: None)
    assert json.loads(out.read_text())["decoding_settings"] == canonical


def test_live_chat_matches_probe_script_output(client):
    import nanollama.compare as cmp
    from nanollama.kb import PERSONAS

    c, engine, _ = client
    system = PERSONAS["assistant"]
    history = [("user", "Hi"), ("assistant", "Hello! How can I help you today?")]
    for settings, extra in ((cmp.GREEDY, {}), (cmp.SAMPLED, {"temperature": cmp.SAMPLED.temperature, "seed": cmp.SAMPLED.seed})):
        events = _stream(c, message="What is Python?", system=system,
                         history=[{"role": r, "content": t} for r, t in history], **extra)
        live = "".join(e["text"] for e in events if e["type"] == "token")
        c.get("/api/health")
        probe = cmp._run(engine.model, engine.tok, system, "What is Python?", history, settings)
        assert live == probe["reply"]


def test_tokenize_reports_normalization(client):
    c, _, _ = client
    t = c.post("/api/tokenize", json={"text": "Hi…"}).json()
    assert t["normalized"] is True and t["total_tokens"] == 5 and t["characters"] == 3
    assert c.post("/api/tokenize", json={"text": "Hi"}).json()["normalized"] is False
