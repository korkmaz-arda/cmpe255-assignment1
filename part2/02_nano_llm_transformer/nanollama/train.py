"""Supervised instruction training from random initialization (F05).

"SFT" here means next-character prediction trained from scratch with the loss
restricted to assistant replies. There is no pretrained base model.

Only the ``train`` and ``val`` splits are opened (via ``open_training_split``).
Final held-out test sets are never touched here; see ``nanollama.finalize``.
"""

from __future__ import annotations

import copy
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import artifacts
from .data.store import SOURCES, open_training_split, read_manifest
from .generate import DecodeSettings, canonical_greedy, reply_text
from .kb import PERSONAS
from .metrics import dense_batches, evaluate_windows, inputs_and_labels, masked_loss_sum, perplexity, unigram_counts, unigram_reference
from .model import ModelConfig, NanoLlama
from .tokenizer import CharTokenizer

SELECTION_SOURCES = ("tinystories", "everyday")
SELECTION_TOLERANCE = 1e-3  # nats; a later epoch must beat the best by more than this to be selected
SELECTION_DEFINITION = (
    "selection_loss = (tinystories_val_loss + everyday_val_loss) / 2. The saved checkpoint is the epoch with the "
    f"lowest selection_loss; a later epoch replaces an earlier one only if it is lower by more than {SELECTION_TOLERANCE} "
    "nats (effective ties keep the earlier epoch). KB validation loss is computed every epoch as a diagnostic but is "
    "not used for selection: KB validation entries contain answers never seen in training."
)

# Fixed prompts that are not items of any dataset split. Outputs are shown as-is.
PROBES = [
    {"kind": "greeting", "persona": "assistant", "message": "Hey there! How is it going?"},
    {"kind": "simple question", "persona": "assistant", "message": "What should I cook for dinner tonight?"},
    {"kind": "storytelling", "persona": "storyteller",
     "message": "Write a short story for young children.\nUse these words: kite, windy, share.\nStory features: Dialogue."},
    {"kind": "out-of-distribution", "persona": "assistant", "message": "What is the capital of Australia?"},
    {"kind": "out-of-distribution", "persona": "python", "message": "Write a Python function that sorts a list of numbers."},
]


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 32
    lr: float = 2e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.05
    grad_clip: float = 1.0
    seed: int = 1234
    everyday_repeat: int = 1  # passes over the Everyday train split per epoch
    kb_repeat: int = 8  # passes over the KB train dialogues per epoch
    max_story_windows: int | None = None  # seeded subsample for smoke runs
    story_windows_per_epoch: int | None = None  # stage 2: fresh seeded TinyStories subset each epoch
    init_from: str | None = None  # stage 2: version directory whose weights initialize training
    compile: bool = False
    tag: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> None:
        if not (1 <= self.epochs <= 100):
            raise ValueError("epochs must be in [1, 100]")
        if not (1 <= self.batch_size <= 256):
            raise ValueError("batch_size must be in [1, 256]")
        if not (1e-5 <= self.lr <= 1e-1):
            raise ValueError("lr must be in [1e-5, 1e-1]")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model"] = self.model.to_dict()
        return d


def improves(selection_loss: float, best_loss: float) -> bool:
    """True if this epoch should replace the saved one; effective ties keep the earlier epoch."""
    return selection_loss < best_loss - SELECTION_TOLERANCE


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    warm = max(1, int(round(cfg.warmup_frac * total)))
    if step < warm:
        return cfg.lr * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    floor = cfg.lr * cfg.min_lr_frac
    return floor + 0.5 * (cfg.lr - floor) * (1 + math.cos(math.pi * min(1.0, progress)))


def device_info(device: torch.device) -> dict:
    info = {"device": str(device), "torch": torch.__version__, "python": platform.python_version()}
    if device.type == "cuda":
        info.update({
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "arch_list": torch.cuda.get_arch_list(),
            "capability": list(torch.cuda.get_device_capability(device)),
        })
    return info


def mixture_plan(train_sets: dict, cfg: TrainConfig) -> dict:
    """Deterministic per-epoch repeat factors for the small sources (training split only)."""
    tokens = {s: train_sets[s].target_tokens for s in SOURCES}
    repeats = {"tinystories": 1, "everyday": cfg.everyday_repeat, "kb": cfg.kb_repeat}
    eff = {s: tokens[s] * repeats[s] for s in SOURCES}
    n_story = len(train_sets["tinystories"])
    if cfg.story_windows_per_epoch and cfg.story_windows_per_epoch < n_story:
        # expected target characters of a random subset of story windows
        eff["tinystories"] = int(round(tokens["tinystories"] * cfg.story_windows_per_epoch / n_story))
    total = sum(eff.values())
    return {
        "repeats": repeats,
        "target_tokens_unique": tokens,
        "target_tokens_per_epoch": eff,
        "effective_share": {s: eff[s] / total for s in SOURCES},
        "story_windows_per_epoch": cfg.story_windows_per_epoch,
        "note": "repeats apply to the training split only"
                + ("; TinyStories uses a fresh seeded subset of training windows each epoch (shares are expected values)"
                   if cfg.story_windows_per_epoch else ""),
    }


def _subsample(ws, owners_keep: int, seed: int):
    from .data.store import WindowSet

    if len(ws) <= owners_keep:
        return ws
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ws), size=owners_keep, replace=False))
    toks, flags, lens = [], [], []
    for i in idx:
        t, f = ws.get(int(i))
        toks.append(t)
        flags.append(f)
        lens.append(len(t))
    offsets = np.zeros(len(idx) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lens)
    return WindowSet(np.concatenate(toks), np.concatenate(flags), offsets, [ws.owners[int(i)] for i in idx])


def run_training(data_root: Path, artifacts_root: Path, cfg: TrainConfig,
                 progress: Callable[[dict], None] | None = None,
                 device: torch.device | str | None = None, promote: bool = True,
                 log: Callable[[str], None] = print) -> Path:
    """Train, write a new version directory, and (by default) promote it. Returns the version dir."""
    cfg.validate()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    manifest = read_manifest(data_root)
    if manifest["context_length"] != cfg.model.context_length:
        raise ValueError(f"prepared data uses context {manifest['context_length']}, model expects {cfg.model.context_length}")

    tok = CharTokenizer.build()
    cfg.model.vocab_size = tok.vocab_size
    train_sets, val_sets, val_dialogues = {}, {}, {}
    for s in SOURCES:
        train_sets[s], _ = open_training_split(data_root, "train", s)
        val_sets[s], val_dialogues[s] = open_training_split(data_root, "val", s)
    if cfg.max_story_windows:
        train_sets["tinystories"] = _subsample(train_sets["tinystories"], cfg.max_story_windows, cfg.seed)
    plan = mixture_plan(train_sets, cfg)
    log(f"mixture: {plan['repeats']} effective share {({k: round(v, 3) for k, v in plan['effective_share'].items()})}")

    length = cfg.model.context_length + 1
    dense_tok, dense_flag, fixed_index = [], [], []
    base = 0
    story_range = (0, 0)
    subsample_stories = bool(cfg.story_windows_per_epoch) and cfg.story_windows_per_epoch < len(train_sets["tinystories"])
    for s in SOURCES:
        t, f = dense_batches(train_sets[s], length)
        dense_tok.append(t)
        dense_flag.append(f)
        if s == "tinystories" and subsample_stories:
            story_range = (base, base + len(t))
        else:
            fixed_index += list(range(base, base + len(t))) * plan["repeats"][s]
        base += len(t)
    all_tok = torch.from_numpy(np.concatenate(dense_tok)).to(device)
    all_flag = torch.from_numpy(np.concatenate(dense_flag)).to(device)
    window_len = torch.from_numpy(np.concatenate([np.diff(train_sets[s].offsets) for s in SOURCES])).to(device)
    fixed_index = np.asarray(fixed_index, dtype=np.int64)
    rng = np.random.default_rng(cfg.seed)

    def epoch_index() -> np.ndarray:
        if not subsample_stories:
            return fixed_index
        size = cfg.story_windows_per_epoch
        stories = story_range[0] + rng.choice(story_range[1] - story_range[0], size=size, replace=False)
        return np.concatenate([fixed_index, stories.astype(np.int64)])

    model = NanoLlama(cfg.model).to(device)
    init_record = None
    if cfg.init_from:
        init_dir = Path(cfg.init_from)
        ckpt = torch.load(init_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
        if ckpt["config"] != cfg.model.to_dict():
            raise ValueError(f"init checkpoint config {ckpt['config']} differs from {cfg.model.to_dict()}; "
                             "the architecture must not change")
        model.load_state_dict(ckpt["state_dict"])
        init_val = {s: evaluate_windows(model, val_sets[s], device, batch_size=max(8, cfg.batch_size)) for s in SOURCES}
        init_record = {"version": init_dir.name, "validation": init_val,
                       "selection_loss": float(np.mean([init_val[s]["loss"] for s in SELECTION_SOURCES]))}
        log(f"initialized from {init_dir.name}: val " + " ".join(f"{s} {init_val[s]['loss']:.4f}" for s in SOURCES)
            + f" | selection {init_record['selection_loss']:.4f}")
    n_params = model.num_parameters()
    log(f"model parameters: {n_params:,} on {device}")
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), fused=device.type == "cuda",
    )
    fwd = torch.compile(model) if (cfg.compile and device.type == "cuda") else model
    use_amp = device.type == "cuda"

    steps_per_epoch = math.ceil((len(fixed_index) + (cfg.story_windows_per_epoch if subsample_stories else 0)) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    curves = {"train_loss": [], "val_loss": {s: [] for s in SOURCES}, "val_selection": [], "lr": [], "grad_norm": [],
              "epoch_seconds": [], "tokens_per_second": []}
    best = {"loss": float("inf"), "epoch": None, "state": None, "val": None}
    step = 0
    t_start = time.time()
    tokens_seen = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t_ep = time.time()
        index = epoch_index()
        order = index[rng.permutation(len(index))]
        loss_sum = torch.zeros((), device=device)
        tok_count = torch.zeros((), device=device, dtype=torch.long)
        gn_sum = torch.zeros((), device=device)
        in_tokens = torch.zeros((), device=device, dtype=torch.long)
        for b in range(steps_per_epoch):
            ids = torch.from_numpy(order[b * cfg.batch_size : (b + 1) * cfg.batch_size]).to(device)
            x, y = inputs_and_labels(all_tok[ids], all_flag[ids])
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, cfg)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits, _ = fwd(x)
            s_loss, count = masked_loss_sum(logits, y)
            loss = s_loss / count.clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            loss_sum += s_loss.detach()
            tok_count += count
            gn_sum += gn.detach()
            in_tokens += (window_len[ids] - 1).sum()
            step += 1
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ep_seconds = time.time() - t_ep
        train_loss = float(loss_sum) / max(1, int(tok_count))
        tokens_seen += int(in_tokens)

        val = {s: evaluate_windows(model, val_sets[s], device, batch_size=max(8, cfg.batch_size)) for s in SOURCES}
        selection = float(np.mean([val[s]["loss"] for s in SELECTION_SOURCES]))
        curves["train_loss"].append(train_loss)
        for s in SOURCES:
            curves["val_loss"][s].append(val[s]["loss"])
        curves["val_selection"].append(selection)
        curves["lr"].append(lr_at(step - 1, total_steps, cfg))
        curves["grad_norm"].append(float(gn_sum) / steps_per_epoch)
        curves["epoch_seconds"].append(ep_seconds)
        curves["tokens_per_second"].append(int(in_tokens) / ep_seconds)
        improved = improves(selection, best["loss"])
        if improved:
            best = {"loss": selection, "epoch": epoch, "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), "val": val}
        event = {
            "epoch": epoch, "epochs": cfg.epochs, "train_loss": train_loss,
            "val_loss": {s: val[s]["loss"] for s in SOURCES}, "val_selection": selection,
            "val_perplexity": {s: val[s]["perplexity"] for s in SOURCES},
            "seconds": ep_seconds, "tokens_per_second": int(in_tokens) / ep_seconds, "best": improved,
        }
        log(f"epoch {epoch}/{cfg.epochs} train {train_loss:.4f} | val " +
            " ".join(f"{s} {val[s]['loss']:.4f}" for s in SOURCES) + f" (diagnostic) | selection {selection:.4f}{' *' if improved else ''}"
            f" | {ep_seconds:.1f}s {event['tokens_per_second']:.0f} tok/s")
        if progress:
            progress(event)

    wall = time.time() - t_start
    vdir = artifacts.new_version_dir(artifacts_root, cfg.tag)
    model.load_state_dict(best["state"])
    model.eval()
    artifacts.save_model(vdir, model, tok, best["state"])

    telemetry = {
        "version": vdir.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "objective": "masked next-character cross-entropy on assistant reply characters and their EOS",
        "training_from": ("random initialization (no pretrained base model)" if not cfg.init_from else
                          f"the earlier trained NanoLlama checkpoint {init_record['version']} "
                          "(stage-2 conversation adaptation; no external pretrained model was ever used)"),
        "stage": 2 if cfg.init_from else 1,
        "initialized_from": init_record,
        "parameters": n_params,
        "config": cfg.to_dict(),
        "optimizer": {"name": "AdamW", "betas": [cfg.beta1, cfg.beta2], "weight_decay": cfg.weight_decay,
                      "weight_decay_applies_to": "matrices and embeddings (not RMSNorm gains)",
                      "grad_clip_norm": cfg.grad_clip,
                      "schedule": f"linear warmup {cfg.warmup_frac:.0%} of steps, cosine decay to {cfg.min_lr_frac:.0%} of peak"},
        "environment": device_info(device),
        "epochs": cfg.epochs, "batch_size": cfg.batch_size, "learning_rate": cfg.lr,
        "steps": total_steps, "steps_per_epoch": steps_per_epoch,
        "curves": curves,
        "selection_definition": SELECTION_DEFINITION,
        "selection_sources": list(SELECTION_SOURCES),
        "diagnostic_only_sources": ["kb"],
        "best_epoch": best["epoch"],
        "best_val_selection_loss": best["loss"],
        "final_train_loss": curves["train_loss"][-1],
        "wall_clock_seconds": wall,
        "input_tokens_seen": tokens_seen,
        "mean_training_tokens_per_second": tokens_seen / max(1e-9, sum(curves["epoch_seconds"])),
        "mixture": plan,
        "perplexity_definition": "exp(mean cross-entropy in nats per character), exponent capped at 20",
    }
    artifacts.write_json(vdir / "telemetry.json", telemetry)
    artifacts.write_json(vdir / "data_report.json", {"manifest": manifest, "mixture": plan,
                                                    "max_story_windows": cfg.max_story_windows})
    artifacts.write_json(vdir / "validation.json", build_validation_report(model, tok, train_sets, val_sets, val_dialogues, best, cfg, device))
    log(f"wrote {vdir}")
    if promote:
        artifacts.promote(artifacts_root, vdir, device)
        log(f"promoted {vdir.name} -> {Path(artifacts_root) / artifacts.POINTER}")
    return vdir


def build_validation_report(model, tok, train_sets, val_sets, val_dialogues, best, cfg: TrainConfig, device) -> dict:
    sources = {}
    for s in SOURCES:
        ref = unigram_reference(unigram_counts(train_sets[s], tok.vocab_size), val_sets[s])
        v = best["val"][s]
        sources[s] = {**v, "unigram_reference": ref}
    greedy = canonical_greedy()
    guarded = DecodeSettings(**{**greedy.to_dict(), "loop_guard": True})
    rng = np.random.default_rng(cfg.seed + 7)
    samples = []
    for s in SOURCES:
        dl = val_dialogues[s]
        for i in sorted(rng.choice(len(dl), size=min(2, len(dl)), replace=False)):
            samples.append(_sample_from_dialogue(model, tok, dl[int(i)], greedy, f"validation:{s}"))
    probes = []
    for p in PROBES:
        probes.append({**p, "decoding": "greedy, loop guard off (default)", "loop_guard": False,
                       "reply": reply_text(model, tok, PERSONAS[p["persona"]], p["message"], greedy),
                       "reply_loop_guard_on": reply_text(model, tok, PERSONAS[p["persona"]], p["message"], guarded)})
    return {
        "split": "validation (used for checkpoint selection; not a final held-out estimate)",
        "best_epoch": best["epoch"],
        "selection_loss": best["loss"],
        "selection_definition": SELECTION_DEFINITION,
        "kb_note": "KB validation loss is a diagnostic of unseen-answer generalization; it is not used for selection.",
        "sources": sources,
        "unigram_reference_definition": "add-one-smoothed character frequencies of this source's training targets, "
                                        "scored on the same validation positions (context-free baseline)",
        "samples": samples,
        "probes": probes,
        "note": "Samples are greedy generations (loop guard off, the default) shown as-is; no quality claim is implied. "
                "Probe replies are also recorded with the loop guard on as a side-by-side diagnostic.",
    }


def dialogue_prompt(d):
    """Split a dialogue into (history, message, reference) at the first substantive exchange.

    For conversations that open with a greeting exchange, the second user turn is
    used so the reply is not just a greeting; the dataset's earlier turns are the history.
    """
    turns = d.turns
    user_idx = [i for i, (r, _) in enumerate(turns) if r == "user"]
    k = user_idx[1] if len(user_idx) > 1 and d.source == "everyday" else user_idx[0]
    ref = turns[k + 1][1] if k + 1 < len(turns) else ""
    return turns[:k], turns[k][1], ref


def _sample_from_dialogue(model, tok, d, settings, label):
    history, message, ref = dialogue_prompt(d)
    reply = reply_text(model, tok, d.system, message, settings, history)
    return {"set": label, "id": d.id, "system": d.system, "history_turns": len(history), "message": message,
            "reference": ref, "reply": reply, "decoding": "greedy, loop guard off", "loop_guard": settings.loop_guard}
