"""Final held-out evaluation of the promoted version — the only code that reads test splits.

Guard rails:
* evaluates the version named by artifacts/current.json and writes final_eval.json
  into that version's directory, so results never carry over to a retrained version;
* refuses if the prepared data no longer matches the manifest the version was trained on
  (a different split could leak test items into training);
* refuses to re-run on an already finalized version unless ``force=True``; forced reruns
  are appended to ``finalize_history`` so repeated looks at the test set are on record.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import artifacts
from .data.store import FINAL_SOURCES, _open_final_split, open_training_split, read_manifest
from .generate import canonical_greedy, reply_text
from .metrics import char_similarity, evaluate_windows, unigram_counts, unigram_reference
from .train import dialogue_prompt

LABELS = {
    "tinystories": "TinyStories held-out test (seeded subset of the official validation file, disjoint from val)",
    "everyday": "Everyday Conversations official test split",
    "kb": "Knowledge-base final-test entries (answers never seen in training)",
    "kb_rephrase": "Knowledge-base rephrasing diagnostic: answer seen during training; wording held out",
}
TRAIN_SOURCE_FOR_REFERENCE = {"tinystories": "tinystories", "everyday": "everyday", "kb": "kb", "kb_rephrase": "kb"}
SIMILARITY_DEFINITION = ("difflib SequenceMatcher ratio between the greedy reply and the reference answer: "
                         "2 x matching characters / total characters, in [0, 1]. Descriptive only: character "
                         "overlap, not semantic correctness.")


class FinalizeRefused(RuntimeError):
    pass


def _manifest_fingerprint(m: dict) -> dict:
    keys = ("seed", "context_length", "tinystories_train_sample", "tinystories_val", "tinystories_test", "everyday_val_fraction")
    return {k: m.get(k) for k in keys} | {"kb_entries": m["report"]["kb"]["entries"]}


def finalize(artifacts_root: Path, data_root: Path, force: bool = False, device: str | None = None,
             samples_per_source: int = 5, log=print) -> Path:
    vdir = artifacts.current_version(artifacts_root)
    if vdir is None:
        raise FinalizeRefused("no promoted version; train one first (python scripts/train.py)")
    out_path = vdir / artifacts.FINAL_EVAL
    previous = artifacts.read_json(out_path)
    if previous is not None and not force:
        raise FinalizeRefused(
            f"version {vdir.name} was already finalized at {previous['finalized_at']}; "
            "re-running looks at the test set again. Pass --force to do it anyway (it will be recorded)."
        )
    trained_on = json.loads((vdir / "data_report.json").read_text())["manifest"]
    current = read_manifest(data_root)
    if _manifest_fingerprint(trained_on) != _manifest_fingerprint(current):
        raise FinalizeRefused("prepared data differs from the data this version was trained on; "
                              "re-run prepare_data.py with the original settings before finalizing")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok, cfg = artifacts.load_model(vdir, device)
    greedy = canonical_greedy()
    rng = np.random.default_rng(4321)
    results = {}
    for source in FINAL_SOURCES:
        ws, dialogues = _open_final_split(data_root, source)
        train_ws, _ = open_training_split(data_root, "train", TRAIN_SOURCE_FOR_REFERENCE[source])
        metrics = evaluate_windows(model, ws, device)
        metrics["unigram_reference"] = unigram_reference(unigram_counts(train_ws, tok.vocab_size), ws)
        if source in ("kb", "kb_rephrase"):
            # one default-persona dialogue per entry, all entries
            chosen, seen = [], set()
            for d in dialogues:
                if d.meta["entry"] not in seen and d.meta.get("persona") == "assistant":
                    seen.add(d.meta["entry"])
                    chosen.append(d)
        else:
            idx = sorted(rng.choice(len(dialogues), size=min(samples_per_source, len(dialogues)), replace=False))
            chosen = [dialogues[int(i)] for i in idx]
        samples = []
        for d in chosen:
            history, message, ref = dialogue_prompt(d)
            reply = reply_text(model, tok, d.system, message, greedy, history)
            item = {"id": d.id, "message": message, "history_turns": len(history), "reference": ref, "reply": reply}
            if source in ("kb", "kb_rephrase"):
                item["similarity"] = char_similarity(reply.strip(), ref.strip())
            samples.append(item)
        if source in ("kb", "kb_rephrase"):
            metrics["mean_similarity"] = float(np.mean([s["similarity"] for s in samples]))
        results[source] = {"label": LABELS[source], **metrics, "samples": samples}
        log(f"{source:12s} loss {metrics['loss']:.4f} ppl {metrics['perplexity']:.3f} "
            f"(unigram ref {metrics['unigram_reference']['loss']:.4f})")

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    history = list(previous.get("finalize_history", [])) if previous else []
    history.append({"at": now, "forced": bool(previous is not None)})
    if previous is not None:
        log(f"WARNING: forced re-finalization of {vdir.name}; this is run #{len(history)} on the test set")
    report = {
        "version": vdir.name,
        "finalized_at": now,
        "finalize_count": len(history),
        "finalize_history": history,
        "split": "final held-out test (not used for training or checkpoint selection)",
        "decoding_for_samples": "canonical greedy default (temperature 0, repetition penalty 1.0, loop guard off, "
                                f"max {greedy.max_new_tokens} new characters) - identical to the live chat default",
        "decoding_settings": greedy.to_dict(),
        "loop_guard_for_samples": False,
        "similarity_definition": SIMILARITY_DEFINITION,
        "unigram_reference_definition": "add-one-smoothed character frequencies of the matching source's training "
                                        "targets, scored on the same test positions (context-free baseline)",
        "sources": results,
    }
    artifacts.write_json(out_path, report)
    log(f"wrote {out_path}")
    return out_path
