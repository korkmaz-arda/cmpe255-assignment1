"""Autoregressive generation with a KV cache, decoding controls and measured timing (F06).

Every character returned here is sampled from NanoLlama's own next-token
distribution. There is no retrieval, prompt matching or canned text.

Logit processing order for each step:
1. mask every control token except EOS (markers can never appear in visible text);
2. repetition penalty over recently generated characters;
3. optional loop guards (off by default): forbid a third identical character in a
   row, and suppress the character that followed an earlier occurrence of the
   current trigram (periodic-loop breaker). The source enabled these by default;
   here they are opt-in because they measurably corrupt ordinary English from the
   trained model (e.g. repeated trigrams such as "the"). They never affect loss metrics;
4. temperature <= 0.15 -> greedy argmax; otherwise temperature, top-k, top-p,
   then a multinomial draw.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterator

import torch

from .model import NanoLlama
from .tokenizer import ASSISTANT, BOS, EOS, SYSTEM, USER, CharTokenizer, normalize_text

GREEDY_TEMPERATURE = 0.15
REPETITION_WINDOW = 32


@dataclass
class DecodeSettings:
    """Decoding controls. The field defaults ARE the canonical configuration (see CANONICAL_GREEDY)."""

    temperature: float = 0.0  # <= GREEDY_TEMPERATURE -> greedy argmax
    top_p: float = 0.9  # used only when sampling
    top_k: int = 40  # used only when sampling
    repetition_penalty: float = 1.0  # 1.0 = off; user-adjustable, never silently applied
    max_new_tokens: int = 300
    loop_guard: bool = False  # optional anti-loop heuristic; off by default (it corrupts valid English)
    seed: int | None = None

    def validate(self) -> None:
        checks = [
            (0.0 <= self.temperature <= 1.5, "temperature must be in [0, 1.5]"),
            (0.1 <= self.top_p <= 1.0, "top_p must be in [0.1, 1.0]"),
            (1 <= self.top_k <= 100, "top_k must be in [1, 100]"),
            (1.0 <= self.repetition_penalty <= 2.0, "repetition_penalty must be in [1.0, 2.0]"),
            (1 <= self.max_new_tokens <= 600, "max_new_tokens must be in [1, 600]"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

    def to_dict(self) -> dict:
        return asdict(self)


def canonical_greedy() -> DecodeSettings:
    """The single default decoding configuration used by live chat, probe comparisons and
    final-evaluation samples: greedy (temperature 0), repetition penalty 1.0, loop guard off.
    It is a direct model-decoding baseline with no heuristic applied."""
    return DecodeSettings()


def canonical_sampled(seed: int = 1234) -> DecodeSettings:
    """The fixed-seed sampled companion: canonical defaults with temperature 0.7 and a seed."""
    return DecodeSettings(temperature=0.7, seed=seed)


def build_prompt_ids(tok: CharTokenizer, system: str, message: str, history: list[tuple[str, str]] | None = None) -> list[int]:
    """BOS + system + history + user message + assistant marker, bodies encoded as plain text."""
    def body(s: str) -> list[int]:
        return tok.encode_plain(normalize_text(s)[0])

    ids = [tok.id_of(BOS), tok.id_of(SYSTEM)] + body(system)
    for role, text in history or []:
        if role == "user":
            ids += [tok.id_of(USER)] + body(text)
        elif role == "assistant":
            ids += [tok.id_of(ASSISTANT)] + body(text) + [tok.id_of(EOS)]
    ids += [tok.id_of(USER)] + body(message) + [tok.id_of(ASSISTANT)]
    return ids


def fit_prompt(tok: CharTokenizer, system: str, message: str, history, context_length: int, max_new_tokens: int):
    """Drop the oldest whole exchanges until the prompt leaves room for the reply.

    Returns (ids, n_history_turns_dropped). Raises ValueError if the system prompt
    plus the current message alone do not fit.
    """
    history = list(history or [])
    dropped = 0
    budget = context_length - max_new_tokens
    while True:
        ids = build_prompt_ids(tok, system, message, history)
        if len(ids) <= budget:
            return ids, dropped
        if not history:
            raise ValueError(
                f"prompt is {len(ids)} characters; with max_new_tokens={max_new_tokens} it must be at most {budget}"
            )
        # remove the oldest user turn and the assistant reply that followed it
        cut = 1
        while cut < len(history) and history[cut][0] != "user":
            cut += 1
        dropped += cut
        history = history[cut:]


def process_logits(logits: torch.Tensor, generated: list[int], tok: CharTokenizer, s: DecodeSettings) -> torch.Tensor:
    """Apply steps 1-3 of the pipeline to a 1-D float logits vector (returns a new tensor)."""
    logits = logits.float().clone()
    eos = tok.eos_id
    for cid in tok.control_ids:
        if cid != eos:
            logits[cid] = float("-inf")

    if s.repetition_penalty != 1.0 and generated:
        recent = torch.tensor(sorted(set(generated[-REPETITION_WINDOW:])), device=logits.device)
        vals = logits[recent]
        logits[recent] = torch.where(vals > 0, vals / s.repetition_penalty, vals * s.repetition_penalty)

    if s.loop_guard and generated:
        if len(generated) >= 2 and generated[-1] == generated[-2]:
            logits[generated[-1]] = float("-inf")
        if len(generated) >= 4:
            tri = generated[-3:]
            for i in range(len(generated) - 4, -1, -1):
                if generated[i : i + 3] == tri:
                    logits[generated[i + 3]] = float("-inf")
                    break
        if torch.isinf(logits).all():  # never leave the distribution empty
            logits[eos] = 0.0
    return logits


def choose_token(logits: torch.Tensor, s: DecodeSettings, generator: torch.Generator | None) -> int:
    if s.temperature <= GREEDY_TEMPERATURE:
        return int(torch.argmax(logits))
    logits = logits / s.temperature
    if s.top_k < logits.numel():
        kth = torch.topk(logits, s.top_k).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
    probs = torch.softmax(logits, dim=-1)
    if s.top_p < 1.0:
        sorted_p, order = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1)
        remove = cum - sorted_p >= s.top_p  # keep the smallest prefix reaching top_p
        sorted_p = sorted_p.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
        probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1, generator=generator))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def stream_generate(model: NanoLlama, tok: CharTokenizer, prompt_ids: list[int], s: DecodeSettings,
                    use_cache: bool = True) -> Iterator[dict]:
    """Yield one event per generated token, then a final ``done`` event.

    Timing is wall-clock with CUDA synchronization. ``use_cache=False`` recomputes
    the full sequence each step (for comparison and tests only).
    """
    s.validate()
    device = next(model.parameters()).device
    ctx = model.cfg.context_length
    if len(prompt_ids) >= ctx:
        raise ValueError(f"prompt of {len(prompt_ids)} tokens leaves no room in context {ctx}")
    # Sampling runs on CPU over the 104-entry logits vector: one device->host copy per
    # step instead of many tiny GPU kernels, and seeded results are device-independent.
    generator = None
    if s.seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(s.seed))

    model.eval()
    t0 = time.perf_counter()
    cache = model.new_cache() if use_cache else None
    seq = list(prompt_ids)
    generated: list[int] = []
    x = torch.tensor([seq], device=device)
    logits, _ = model(x, cache=cache)
    last = logits[0, -1].float().cpu()
    first_token_time = None
    stop_reason = "max_new_tokens"
    for step in range(s.max_new_tokens):
        nxt = choose_token(process_logits(last, generated, tok, s), s, generator)
        now = time.perf_counter()
        if first_token_time is None:
            first_token_time = now - t0
        if nxt == tok.eos_id:
            stop_reason = "eos"
            break
        generated.append(nxt)
        seq.append(nxt)
        elapsed = now - t0
        decode_elapsed = now - t0 - first_token_time
        tps = (len(generated) - 1) / decode_elapsed if len(generated) > 1 and decode_elapsed > 0 else None
        yield {
            "type": "token",
            "id": nxt,
            "text": tok.decode([nxt]),
            "count": len(generated),
            "tokens_per_second": tps,
            "ttft_ms": first_token_time * 1000.0,
            "elapsed_ms": elapsed * 1000.0,
        }
        if len(seq) >= ctx:
            stop_reason = "context_full"
            break
        if use_cache:
            logits, _ = model(torch.tensor([[nxt]], device=device), cache=cache)
        else:
            logits, _ = model(torch.tensor([seq], device=device))
        last = logits[0, -1].float().cpu()
    _sync(device)
    total = time.perf_counter() - t0
    decode_time = total - (first_token_time or 0.0)
    yield {
        "type": "done",
        "count": len(generated),
        "stop_reason": stop_reason,
        "total_ms": total * 1000.0,
        "ttft_ms": (first_token_time or total) * 1000.0,
        "tokens_per_second": (len(generated) / decode_time) if generated and decode_time > 0 else None,
        "kv_cache": use_cache,
        "device": str(device),
    }


def generate(model: NanoLlama, tok: CharTokenizer, prompt_ids: list[int], s: DecodeSettings, use_cache: bool = True) -> tuple[list[int], str]:
    ids, reason = [], ""
    for ev in stream_generate(model, tok, prompt_ids, s, use_cache):
        if ev["type"] == "token":
            ids.append(ev["id"])
        else:
            reason = ev["stop_reason"]
    return ids, reason


def reply_text(model: NanoLlama, tok: CharTokenizer, system: str, message: str, s: DecodeSettings,
               history=None) -> str | None:
    """Generate a reply for reports. If the prompt leaves less room than ``s.max_new_tokens``,
    older history is dropped first, then the reply budget shrinks to what the context allows.
    Returns None only when the prompt alone fills the context."""
    ctx = model.cfg.context_length
    try:
        ids, _ = fit_prompt(tok, system, message, history, ctx, s.max_new_tokens)
        settings = s
    except ValueError:
        ids = build_prompt_ids(tok, system, message)
        room = ctx - len(ids)
        if room < 1:
            return None
        settings = DecodeSettings(**{**s.to_dict(), "max_new_tokens": min(s.max_new_tokens, room)})
    out, _ = generate(model, tok, ids, settings)
    return tok.decode(out, skip_control=True)
