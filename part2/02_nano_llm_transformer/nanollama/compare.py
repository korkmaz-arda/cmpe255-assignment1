"""Side-by-side generation probes for comparing two checkpoints (validation-safe).

Probe prompts are fixed, hand-written, and asserted not to be KB final-test or
rephrasing-diagnostic queries. Everyday prompts come from the *validation* split.
No final test data is read.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from . import kb
from .data.records import norm_key
from .data.store import open_training_split
from .generate import DecodeSettings, canonical_greedy, canonical_sampled, fit_prompt, generate
from .train import dialogue_prompt

FIXED_PROBES = [
    {"kind": "greeting", "persona": "assistant", "message": "Hi!"},
    {"kind": "identity", "persona": "assistant", "message": "What is your name?"},
    {"kind": "everyday help", "persona": "assistant", "message": "Can you help me plan a picnic?"},
    {"kind": "simple question", "persona": "assistant", "message": "What is Python?"},
    {"kind": "story", "persona": "storyteller", "message": "Tell me a short story about a dog."},
    {"kind": "story (assistant persona)", "persona": "assistant", "message": "Tell me a short story about a dog."},
    {"kind": "arithmetic", "persona": "math_tutor", "message": "What is 7 plus 5?"},
    {"kind": "KB-style (train entry)", "persona": "python", "message": "Write a factorial function in Python."},
    {"kind": "KB-style (train entry)", "persona": "ml_research", "message": "What is gradient descent?"},
    {"kind": "out-of-distribution", "persona": "assistant", "message": "Who won the football world cup in 1998?"},
]

# Pre-specified before the adaptation run; applied by a person reading adaptation_comparison.json.
ACCEPTANCE_CRITERIA = [
    "selection loss (TinyStories-val + Everyday-val)/2 decreases by more than 0.001 nats",
    "Everyday validation loss decreases",
    "TinyStories validation loss does not worsen by more than 0.02 nats",
    "looping replies (greedy + sampled) do not increase",
    "reading the fixed probes, conversational replies (greeting, identity, picnic, Python, Everyday validation prompts) "
    "are clearly more on-topic overall, and story probes keep their fluency; judged across all probes, not one prompt",
]

GREEDY = canonical_greedy()  # same object definition the chat API/UI default to
SAMPLED = canonical_sampled(1234)


def assert_validation_safe() -> None:
    parts = kb.build_kb_splits()
    held = {norm_key(d.turns[0][1]) for role in ("test", "rephrase") for d in parts[role]}
    for p in FIXED_PROBES:
        if norm_key(p["message"]) in held:
            raise AssertionError(f"probe {p['message']!r} is a KB final-test/diagnostic query")


def loop_stats(text: str, n: int = 12) -> dict:
    """Repetition diagnostics: most frequent n-character substring count and distinct n-gram ratio."""
    grams = [text[i : i + n] for i in range(max(0, len(text) - n + 1))]
    if not grams:
        return {"max_repeat_12gram": 0, "distinct_12gram_ratio": 1.0, "looping": False}
    counts = Counter(grams)
    top = counts.most_common(1)[0][1]
    return {"max_repeat_12gram": top, "distinct_12gram_ratio": len(counts) / len(grams), "looping": top >= 3}


def _run(model, tok, system, message, history, settings):
    ids, _ = fit_prompt(tok, system, message, history, model.cfg.context_length, settings.max_new_tokens)
    out, reason = generate(model, tok, ids, settings)
    text = tok.decode(out, skip_control=True)
    return {"reply": text, "stop_reason": reason, "chars": len(text), **loop_stats(text)}


def probe_set(val_everyday_dialogues: list, n_everyday: int = 3, seed: int = 99) -> list[dict]:
    assert_validation_safe()
    items = [{**p, "system": kb.PERSONAS[p["persona"]], "history": [], "source": "fixed probe"} for p in FIXED_PROBES]
    rng = np.random.default_rng(seed)
    for i in sorted(rng.choice(len(val_everyday_dialogues), size=n_everyday, replace=False)):
        d = val_everyday_dialogues[int(i)]
        history, message, ref = dialogue_prompt(d)
        items.append({"kind": "everyday validation", "persona": "assistant", "message": message, "system": d.system,
                      "history": history, "reference": ref, "source": f"validation:{d.id}"})
    return items


def compare(models: dict, tok, data_root) -> dict:
    """models: {label: model}. Returns probes with greedy and fixed-seed sampled replies per model."""
    _, val_everyday = open_training_split(data_root, "val", "everyday")
    probes = probe_set(val_everyday)
    rows = []
    for p in probes:
        row = {k: p[k] for k in ("kind", "persona", "message", "source")}
        row["history_turns"] = len(p["history"])
        if "reference" in p:
            row["reference"] = p["reference"]
        for label, model in models.items():
            row[label] = {
                "greedy": _run(model, tok, p["system"], p["message"], p["history"], GREEDY),
                "sampled_seed1234": _run(model, tok, p["system"], p["message"], p["history"], SAMPLED),
            }
        rows.append(row)
    summary = {}
    for label in models:
        replies = [r[label][mode] for r in rows for mode in ("greedy", "sampled_seed1234")]
        summary[label] = {
            "replies": len(replies),
            "looping_replies": sum(r["looping"] for r in replies),
            "ended_with_eos": sum(r["stop_reason"] == "eos" for r in replies),
            "mean_chars": float(np.mean([r["chars"] for r in replies])),
        }
    return {
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
        "decoding": {"greedy": GREEDY.to_dict(), "sampled": SAMPLED.to_dict()},
        "loop_definition": "looping = some 12-character substring occurs 3 or more times in the reply",
        "summary": summary,
        "probes": rows,
    }
