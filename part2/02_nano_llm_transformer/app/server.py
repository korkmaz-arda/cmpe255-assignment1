"""FastAPI server for the NanoLlama studio (single page + JSON/NDJSON endpoints).

Run (localhost only):
    uvicorn app.server:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine import Engine, ModelUnavailable  # noqa: E402
from nanollama.generate import DecodeSettings, canonical_greedy, canonical_sampled  # noqa: E402

_DEFAULT = canonical_greedy()

STATIC = Path(__file__).parent / "static"


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    system: str = Field(default="You are NanoLlama, a small helpful assistant.", max_length=1000)
    history: list[Turn] = Field(default_factory=list)
    # Defaults come from the canonical decoding configuration shared with probes and finalization.
    temperature: float = _DEFAULT.temperature
    top_p: float = _DEFAULT.top_p
    top_k: int = _DEFAULT.top_k
    repetition_penalty: float = _DEFAULT.repetition_penalty
    max_new_tokens: int = _DEFAULT.max_new_tokens
    loop_guard: bool = _DEFAULT.loop_guard
    seed: int | None = _DEFAULT.seed


class TextRequest(BaseModel):
    text: str = Field(max_length=4000)


class RetrainRequest(BaseModel):
    epochs: int
    batch_size: int
    lr: float
    budget: str = "quick"


def create_app(engine: Engine | None = None) -> FastAPI:
    engine = engine or Engine()
    app = FastAPI(title="NanoLlama Studio", docs_url=None, redoc_url=None)
    app.state.engine = engine

    def guarded(fn, *args):
        try:
            return fn(*args)
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return engine.health()

    @app.get("/api/decoding-defaults")
    def decoding_defaults():
        return {"default": canonical_greedy().to_dict(), "sampled_probe": canonical_sampled().to_dict(),
                "note": "The chat UI initializes its controls from 'default'; probes and final-evaluation samples use the same."}

    @app.get("/api/presets")
    def presets():
        return engine.presets()

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        settings = DecodeSettings(temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                                  repetition_penalty=req.repetition_penalty, max_new_tokens=req.max_new_tokens,
                                  loop_guard=req.loop_guard, seed=req.seed)
        history = [(t.role, t.content) for t in req.history]
        try:
            settings.validate()
            gen = engine.chat_stream(req.system, req.message, history, settings)
            first = next(gen)  # surfaces load/validation errors as HTTP errors before streaming starts
        except ModelUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        def body():
            yield json.dumps(first) + "\n"
            try:
                for event in gen:
                    yield json.dumps(event) + "\n"
            except Exception as exc:  # stream ends cleanly with an error event
                yield json.dumps({"type": "error", "message": f"{type(exc).__name__}: {exc}"}) + "\n"

        return StreamingResponse(body(), media_type="application/x-ndjson")

    @app.post("/api/tokenize")
    def tokenize(req: TextRequest):
        return guarded(engine.tokenize, req.text)

    @app.get("/api/control-tokens")
    def control_tokens():
        return engine.control_tokens()

    @app.post("/api/attention")
    def attention(req: TextRequest):
        if not req.text.strip():
            raise HTTPException(status_code=422, detail="enter a prompt to inspect")
        return guarded(engine.attention, req.text)

    @app.get("/api/telemetry")
    def telemetry():
        return engine.telemetry()

    @app.get("/api/architecture")
    def architecture():
        return engine.architecture()

    @app.post("/api/retrain")
    def retrain(req: RetrainRequest):
        try:
            engine.start_retrain(req.epochs, req.batch_size, req.lr, req.budget)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return engine.job_status()

    @app.get("/api/retrain")
    def retrain_status():
        return engine.job_status()

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()
