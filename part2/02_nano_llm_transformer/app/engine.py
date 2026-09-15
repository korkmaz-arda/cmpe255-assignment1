"""Application state: one shared model holder, lazy loading, and the retraining job.

Framework-independent logic lives in ``nanollama``; this module only manages
process-wide state (the promoted version currently served, locks, a background
retraining thread) for the FastAPI layer.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

from nanollama import artifacts
from nanollama.generate import DecodeSettings, fit_prompt, stream_generate
from nanollama.kb import PERSONA_LABELS, PERSONAS, presets
from nanollama.metrics import perplexity
from nanollama.model import expected_param_count
from nanollama.tokenizer import CONTROL_DESCRIPTIONS, CONTROL_TOKENS, normalize_text
from nanollama.train import TrainConfig, run_training

ROOT = Path(__file__).resolve().parents[1]
ATTENTION_MAX_TOKENS = 64
TOP_K_PREDICTIONS = 5

# Server-side retraining bounds (the UI sliders sit inside these).
RETRAIN_BOUNDS = {"epochs": (1, 20), "batch_size": (4, 64), "lr": (1e-4, 1e-2)}
RETRAIN_BUDGETS = {"quick": 12000, "full": None}  # story windows used; None = all prepared training windows


@dataclass
class RetrainJob:
    status: str = "idle"  # idle | running | succeeded | failed
    params: dict = field(default_factory=dict)
    epochs: list = field(default_factory=list)
    started_at: float | None = None
    finished_at: float | None = None
    version: str | None = None
    error: str | None = None


class Engine:
    def __init__(self, artifacts_root: Path = ROOT / "artifacts", data_root: Path = ROOT / "data" / "processed",
                 device: str | None = None):
        self.artifacts_root = Path(artifacts_root)
        self.data_root = Path(data_root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.tok = None
        self.version_dir: Path | None = None
        self.load_error: str | None = None
        self._model_lock = threading.RLock()
        self._job_lock = threading.Lock()
        self.job = RetrainJob()

    # ------------------------------------------------------------------ loading
    def load_current(self) -> bool:
        """Load the version named by current.json (S07 lazy recovery calls this on demand)."""
        with self._model_lock:
            vdir = artifacts.current_version(self.artifacts_root)
            if vdir is None:
                self.load_error = "No promoted model version yet. Run: python scripts/train.py"
                return False
            if self.model is not None and self.version_dir == vdir:
                return True
            try:
                model, tok, _ = artifacts.load_model(vdir, self.device)
            except Exception as exc:  # keep serving whatever was loaded before
                self.load_error = f"Failed to load {vdir.name}: {exc}"
                return self.model is not None
            self.model, self.tok, self.version_dir = model, tok, vdir
            self.load_error = None
            return True

    def require_model(self):
        if not self.load_current():
            raise ModelUnavailable(self.load_error or "model not loaded")
        return self.model, self.tok

    # ------------------------------------------------------------------- health
    def health(self) -> dict:
        loaded = self.load_current()
        out = {"model_loaded": loaded, "device": str(self.device), "error": self.load_error}
        if loaded:
            out.update({
                "version": self.version_dir.name,
                "parameters": self.model.num_parameters(),
                "vocab_size": self.tok.vocab_size,
                "context_length": self.model.cfg.context_length,
                "kv_cache": True,
                "finalized": (self.version_dir / artifacts.FINAL_EVAL).exists(),
            })
        return out

    # --------------------------------------------------------------------- chat
    def chat_stream(self, system: str, message: str, history: list, settings: DecodeSettings):
        model, tok = self.require_model()
        settings.validate()
        ids, dropped = fit_prompt(tok, system, message, history, model.cfg.context_length, settings.max_new_tokens)
        yield {"type": "start", "prompt_tokens": len(ids), "history_turns_dropped": dropped,
               "version": self.version_dir.name, "settings": settings.to_dict()}
        with self._model_lock:
            yield from stream_generate(model, tok, ids, settings, use_cache=True)

    # ---------------------------------------------------------------- tokenizer
    def tokenize(self, text: str) -> dict:
        model, tok = self.require_model()
        norm, replaced = normalize_text(text)
        ids = tok.encode_plain(norm)
        tokens = [{"id": i, "text": tok.tokens[i]} for i in ids]
        predictions = []
        ctx_ids = ([tok.bos_id] + ids)[-model.cfg.context_length:]
        with self._model_lock, torch.no_grad():
            logits, _ = model(torch.tensor([ctx_ids], device=self.device))
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
            top = torch.topk(probs, TOP_K_PREDICTIONS)
        for p, i in zip(top.values.tolist(), top.indices.tolist()):
            predictions.append({"id": i, "text": tok.tokens[i], "percent": 100.0 * p, "is_control": i < len(CONTROL_TOKENS)})
        return {
            "tokens": tokens,
            "total_tokens": len(ids),
            "characters": len(text),
            "chars_per_token": (len(text) / len(ids)) if ids else None,
            "replaced_with_space": replaced,
            "normalized": norm != text,
            "top_predictions": predictions,
            "context_note": "Prediction for the character after the text, given BOS + the text (no chat template).",
        }

    def control_tokens(self) -> list[dict]:
        return [{"id": i, "token": t, "description": CONTROL_DESCRIPTIONS[t]} for i, t in enumerate(CONTROL_TOKENS)]

    # ---------------------------------------------------------------- attention
    def attention(self, text: str) -> dict:
        model, tok = self.require_model()
        ids = [tok.bos_id] + tok.encode_plain(normalize_text(text)[0])
        truncated = len(ids) > ATTENTION_MAX_TOKENS
        ids = ids[:ATTENTION_MAX_TOKENS]
        with self._model_lock, torch.no_grad():
            _, attn = model(torch.tensor([ids], device=self.device), return_attention=True)
        layers = [[[[round(v, 5) for v in row] for row in head] for head in layer[0].float().cpu().tolist()] for layer in attn]
        return {
            "tokens": [tok.tokens[i] for i in ids],
            "n_layers": len(layers),
            "n_heads": len(layers[0]),
            "size": len(ids),
            "truncated": truncated,
            "max_tokens": ATTENTION_MAX_TOKENS,
            "attention": layers,  # [layer][head][query][key]
        }

    # ---------------------------------------------------------------- telemetry
    def telemetry(self) -> dict:
        loaded = self.load_current()
        if not loaded:
            return {"available": False, "reason": self.load_error}
        vdir = self.version_dir
        tel = artifacts.read_json(vdir / "telemetry.json")
        val = artifacts.read_json(vdir / "validation.json")
        final = artifacts.read_json(vdir / artifacts.FINAL_EVAL)
        promo = artifacts.read_json(vdir / "promotion.json")
        report = artifacts.read_json(vdir / "data_report.json")
        cfg = tel["config"]["model"]
        best = tel["best_epoch"] - 1
        val_sel = {s: tel["curves"]["val_loss"][s][best] for s in tel["curves"]["val_loss"]}
        return {
            "available": True,
            "version": vdir.name,
            "promotion": promo,
            "model_config": cfg,
            "parameters": tel["parameters"],
            "parameters_formula": expected_param_count(self.model.cfg),
            "training": {k: tel[k] for k in ("objective", "training_from", "optimizer", "environment", "epochs", "batch_size",
                                             "learning_rate", "steps", "best_epoch", "best_val_selection_loss",
                                             "final_train_loss", "wall_clock_seconds", "input_tokens_seen",
                                             "mean_training_tokens_per_second", "selection_definition", "selection_sources",
                                             "perplexity_definition", "created", "mixture")},
            "curves": tel["curves"],
            "selected_validation": {
                "selection_loss": tel["best_val_selection_loss"],
                "selection_perplexity": perplexity(tel["best_val_selection_loss"]),
                "per_source_loss": val_sel,
            },
            "validation": val,
            "data_report": report,
            "final": final,  # None until scripts/finalize.py runs on this version
            "finalized": final is not None,
        }

    def architecture(self) -> dict:
        loaded = self.load_current()
        if not loaded:
            return {"available": False, "reason": self.load_error}
        c = self.model.cfg
        return {"available": True, "version": self.version_dir.name, "config": c.to_dict(), "head_dim": c.head_dim,
                "parameters": self.model.num_parameters()}

    def presets(self) -> dict:
        return {"presets": presets(), "personas": [{"key": k, "label": PERSONA_LABELS[k], "system": v} for k, v in PERSONAS.items()]}

    # ---------------------------------------------------------------- retraining
    def start_retrain(self, epochs: int, batch_size: int, lr: float, budget: str) -> RetrainJob:
        errors = []
        for name, value in (("epochs", epochs), ("batch_size", batch_size), ("lr", lr)):
            lo, hi = RETRAIN_BOUNDS[name]
            if not (lo <= value <= hi):
                errors.append(f"{name} must be between {lo} and {hi}")
        if budget not in RETRAIN_BUDGETS:
            errors.append(f"budget must be one of {sorted(RETRAIN_BUDGETS)}")
        if errors:
            raise ValueError("; ".join(errors))
        with self._job_lock:
            if self.job.status == "running":
                raise RuntimeError("a retraining run is already in progress")
            self.job = RetrainJob(status="running", params={"epochs": epochs, "batch_size": batch_size, "lr": lr, "budget": budget},
                                  started_at=time.time())
        cfg = TrainConfig(epochs=epochs, batch_size=batch_size, lr=lr, max_story_windows=RETRAIN_BUDGETS[budget],
                          compile=self.device.type == "cuda", tag=f"retrain-{budget}")
        threading.Thread(target=self._retrain_worker, args=(cfg,), daemon=True).start()
        return self.job

    def _retrain_worker(self, cfg: TrainConfig) -> None:
        job = self.job
        try:
            # run_training writes a new version dir, verifies, probes, then swaps current.json.
            vdir = run_training(self.data_root, self.artifacts_root, cfg, progress=job.epochs.append,
                                device=self.device, promote=True, log=lambda *_: None)
            # current.json now names the new version; load_current swaps it in under the lock.
            # Until that swap completes, requests keep using the previous model object.
            if not self.load_current() or self.version_dir != vdir:
                raise RuntimeError(self.load_error or "promoted version could not be loaded")
            job.version = vdir.name
            job.status = "succeeded"
        except Exception as exc:
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
            traceback.print_exc()
            self.load_current()  # keep (or restore) the previously promoted version
        finally:
            job.finished_at = time.time()

    def job_status(self) -> dict:
        j = self.job
        return {"status": j.status, "params": j.params, "epochs": j.epochs, "version": j.version, "error": j.error,
                "started_at": j.started_at, "finished_at": j.finished_at,
                "elapsed_seconds": ((j.finished_at or time.time()) - j.started_at) if j.started_at else None,
                "bounds": RETRAIN_BOUNDS, "budgets": {k: v for k, v in RETRAIN_BUDGETS.items()}}


class ModelUnavailable(RuntimeError):
    pass
