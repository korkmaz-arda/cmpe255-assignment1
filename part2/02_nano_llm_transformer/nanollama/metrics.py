"""Split-agnostic evaluation helpers: masked cross-entropy, perplexity, unigram reference.

These functions never open files; callers decide which split they may pass in.
"""

from __future__ import annotations

import difflib
import math

import numpy as np
import torch
import torch.nn.functional as F

from .data.store import WindowSet
from .tokenizer import IGNORE_INDEX

PPL_EXP_CAP = 20.0  # perplexity = exp(min(loss, cap)) to avoid overflow in early epochs


def perplexity(loss: float) -> float:
    return math.exp(min(loss, PPL_EXP_CAP))


def dense_batches(ws: WindowSet, length: int, pad_id: int = 0):
    """Materialize windows as padded [N, length] token and label arrays (numpy)."""
    n = len(ws)
    tokens = np.full((n, length), pad_id, dtype=np.uint8)
    flags = np.zeros((n, length), dtype=bool)
    for i in range(n):
        t, f = ws.get(i)
        tokens[i, : len(t)] = t
        flags[i, : len(f)] = f
    return tokens, flags


def inputs_and_labels(tokens: torch.Tensor, flags: torch.Tensor):
    """tokens/flags: [B, L+1] -> inputs [B, L] (long), labels [B, L] with IGNORE where not a target."""
    x = tokens[:, :-1].long()
    y = tokens[:, 1:].long()
    y = torch.where(flags[:, 1:], y, torch.full_like(y, IGNORE_INDEX))
    return x, y


def masked_loss_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=IGNORE_INDEX, reduction="sum")
    return loss, (labels != IGNORE_INDEX).sum()


@torch.no_grad()
def evaluate_windows(model, ws: WindowSet, device: torch.device, batch_size: int = 32, amp: bool = True) -> dict:
    """Token-weighted mean masked cross-entropy over all supervised positions."""
    if len(ws) == 0:
        return {"loss": None, "perplexity": None, "target_tokens": 0, "windows": 0}
    model.eval()
    length = int(np.diff(ws.offsets).max())
    tokens, flags = dense_batches(ws, length)
    total, count = 0.0, 0
    use_amp = amp and device.type == "cuda"
    for i in range(0, len(ws), batch_size):
        t = torch.from_numpy(tokens[i : i + batch_size]).to(device)
        f = torch.from_numpy(flags[i : i + batch_size]).to(device)
        x, y = inputs_and_labels(t, f)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            logits, _ = model(x)
        s, c = masked_loss_sum(logits, y)
        total += float(s)
        count += int(c)
    loss = total / max(1, count)
    return {"loss": loss, "perplexity": perplexity(loss), "target_tokens": count, "windows": len(ws)}


def unigram_counts(ws: WindowSet, vocab_size: int) -> np.ndarray:
    """Character frequencies of supervised label positions (position 0 of a window is never a label)."""
    mask = ws.targets.copy()
    mask[ws.offsets[:-1]] = False
    return np.bincount(ws.tokens[mask], minlength=vocab_size).astype(np.float64)


def unigram_reference(train_counts: np.ndarray, eval_ws: WindowSet) -> dict:
    """Cross-entropy of a add-one-smoothed unigram character model fit on training targets.

    A context-free baseline: it predicts every character from its overall training
    frequency. A trained transformer should score well below it.
    """
    probs = (train_counts + 1.0) / (train_counts.sum() + len(train_counts))
    eval_counts = unigram_counts(eval_ws, len(train_counts))
    n = eval_counts.sum()
    if n == 0:
        return {"loss": None, "perplexity": None}
    loss = float(-(eval_counts * np.log(probs)).sum() / n)
    return {"loss": loss, "perplexity": perplexity(loss)}


def char_similarity(a: str, b: str) -> float:
    """difflib SequenceMatcher ratio in [0, 1]: 2 * matched characters / total characters."""
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
