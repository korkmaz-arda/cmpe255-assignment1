"""Versioned model artifacts and atomic promotion.

    artifacts/versions/<run-id>/  checkpoint.pt, config.json, vocab.json, telemetry.json,
                                  validation.json, data_report.json, promotion.json,
                                  [final_eval.json]  (added only by scripts/finalize.py)
    artifacts/current.json        {"version": "<run-id>", "promoted_at": ...}

Promotion: verify files -> reload from disk -> probe inference (finite) -> write
promotion.json -> atomically replace current.json. Any failure leaves
current.json (and therefore the served model) untouched.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch

from .model import ModelConfig, NanoLlama
from .tokenizer import CharTokenizer

REQUIRED_FILES = ("checkpoint.pt", "config.json", "vocab.json", "telemetry.json", "validation.json", "data_report.json")
POINTER = "current.json"
HISTORY = "promotion_history.jsonl"
FINAL_EVAL = "final_eval.json"


class PromotionError(RuntimeError):
    pass


def versions_dir(root: Path) -> Path:
    return Path(root) / "versions"


def new_version_dir(root: Path, tag: str = "run") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = versions_dir(root) / f"{stamp}-{tag}"
    d, i = base, 1
    while d.exists():
        i += 1
        d = base.with_name(f"{base.name}-{i}")
    d.mkdir(parents=True)
    return d


def write_json(path: Path, obj) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def save_model(vdir: Path, model: NanoLlama, tok: CharTokenizer, state_dict: dict | None = None) -> None:
    sd = state_dict if state_dict is not None else model.state_dict()
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in sd.items()}, "config": model.cfg.to_dict()},
               Path(vdir) / "checkpoint.pt")
    write_json(Path(vdir) / "config.json", model.cfg.to_dict())
    tok.save(Path(vdir) / "vocab.json")


def load_model(vdir: Path, device: torch.device | str = "cpu") -> tuple[NanoLlama, CharTokenizer, ModelConfig]:
    """Rebuild the network from the config stored with the checkpoint."""
    vdir = Path(vdir)
    ckpt = torch.load(vdir / "checkpoint.pt", map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = NanoLlama(cfg)
    model.load_state_dict(ckpt["state_dict"])
    tok = CharTokenizer.load(vdir / "vocab.json")
    if tok.vocab_size != cfg.vocab_size:
        raise PromotionError(f"vocab size {tok.vocab_size} does not match model config {cfg.vocab_size}")
    return model.to(device).eval(), tok, cfg


def probe(model: NanoLlama, tok: CharTokenizer) -> dict:
    """Fixed probe: a forward pass and a short greedy generation must yield finite logits."""
    from .generate import DecodeSettings, build_prompt_ids, generate
    from .kb import PERSONAS

    device = next(model.parameters()).device
    ids = build_prompt_ids(tok, PERSONAS["assistant"], "Hello!")
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids], device=device))
    if not torch.isfinite(logits).all():
        raise PromotionError("probe forward pass produced non-finite logits")
    out, reason = generate(model, tok, ids, DecodeSettings(temperature=0.0, max_new_tokens=16, loop_guard=False))
    return {"prompt_chars": len(ids), "generated_chars": len(out), "stop_reason": reason, "finite_logits": True}


def verify_version(vdir: Path, device: torch.device | str = "cpu") -> dict:
    vdir = Path(vdir)
    missing = [f for f in REQUIRED_FILES if not (vdir / f).exists()]
    if missing:
        raise PromotionError(f"version {vdir.name} is missing {missing}")
    for f in REQUIRED_FILES:
        if f.endswith(".json"):
            json.loads((vdir / f).read_text())
    model, tok, _ = load_model(vdir, device)
    result = probe(model, tok)
    tel = json.loads((vdir / "telemetry.json").read_text())
    if tel.get("parameters") != model.num_parameters():
        raise PromotionError("telemetry parameter count does not match the checkpoint")
    for k in ("best_val_selection_loss",):
        v = tel.get(k)
        if v is None or not math.isfinite(v):
            raise PromotionError(f"telemetry {k} is missing or not finite")
    return result


def _append_history(root: Path, entry: dict) -> None:
    """Every promotion attempt is logged outside the (immutable) version directories."""
    with open(Path(root) / HISTORY, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def promote(root: Path, vdir: Path, device: torch.device | str = "cpu") -> dict:
    """Verify, probe, then atomically point current.json at ``vdir``.

    promotion.json records a version's first promotion (or its failure) and is never
    rewritten once the version has been promoted, so re-promoting an older version (for
    example restoring it after a retrain) does not modify its directory. Every attempt is
    appended to artifacts/promotion_history.jsonl.
    """
    root, vdir = Path(root), Path(vdir)
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    existing = read_json(vdir / "promotion.json") if (vdir / "promotion.json").exists() else None
    already_promoted = bool(existing) and existing.get("status") == "promoted"
    try:
        probe_result = verify_version(vdir, device)
    except Exception as exc:
        if not already_promoted:
            write_json(vdir / "promotion.json", {"status": "failed", "error": str(exc), "at": now})
        _append_history(root, {"version": vdir.name, "status": "failed", "error": str(exc), "at": now})
        raise PromotionError(str(exc)) from exc
    record = {"status": "promoted", "probe": probe_result, "at": now}
    if not already_promoted:
        write_json(vdir / "promotion.json", record)
    write_json(root / POINTER, {"version": vdir.name, "promoted_at": now})
    _append_history(root, {"version": vdir.name, "status": "promoted", "repromotion": already_promoted, "at": now})
    return record


def current_version(root: Path) -> Path | None:
    p = Path(root) / POINTER
    if not p.exists():
        return None
    name = json.loads(p.read_text())["version"]
    vdir = versions_dir(root) / name
    return vdir if vdir.exists() else None


def read_json(path: Path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None
