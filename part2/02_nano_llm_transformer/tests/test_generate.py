import pytest
import torch

from nanollama.generate import DecodeSettings, build_prompt_ids, choose_token, fit_prompt, generate, process_logits, stream_generate
from nanollama.model import NanoLlama
from nanollama.tokenizer import CharTokenizer
from tests.conftest import tiny_model_config

TOK = CharTokenizer.build()


def _model(ctx=256):
    torch.manual_seed(3)
    return NanoLlama(tiny_model_config(ctx)).eval()


def test_control_tokens_except_eos_are_masked():
    logits = torch.zeros(104)
    logits[3] = 50.0  # <|user|> would win
    out = process_logits(logits, [], TOK, DecodeSettings(loop_guard=False, repetition_penalty=1.0))
    assert torch.isinf(out[[0, 1, 3, 4, 5]]).all() and torch.isfinite(out[2])


def test_repetition_penalty_changes_logits_of_recent_characters():
    logits = torch.linspace(-2, 2, 104)
    gen = [TOK.id_of("a"), TOK.id_of("~")]
    base = process_logits(logits, gen, TOK, DecodeSettings(loop_guard=False, repetition_penalty=1.0))
    pen = process_logits(logits, gen, TOK, DecodeSettings(loop_guard=False, repetition_penalty=1.5))
    for i in gen:
        assert pen[i] < base[i]
    others = [i for i in range(6, 104) if i not in gen]
    assert torch.equal(pen[others], base[others])


def test_loop_guards():
    s = DecodeSettings(loop_guard=True, repetition_penalty=1.0)
    a, b, c, d = (TOK.id_of(ch) for ch in "abcd")
    out = process_logits(torch.zeros(104), [a, a], TOK, s)
    assert torch.isinf(out[a])  # no third identical character in a row
    out = process_logits(torch.zeros(104), [a, b, c, d, a, b, c], TOK, s)
    assert torch.isinf(out[d])  # trigram "abc" was followed by "d" before
    off = process_logits(torch.zeros(104), [a, b, c, d, a, b, c], TOK, DecodeSettings(loop_guard=False, repetition_penalty=1.0))
    assert torch.isfinite(off[d])


def test_greedy_threshold_top_k_and_top_p():
    logits = torch.tensor([0.0] * 100 + [1.0, 3.0, 2.0, 0.5])
    g = torch.Generator().manual_seed(0)
    assert choose_token(logits, DecodeSettings(temperature=0.15), g) == 101
    for _ in range(20):
        assert choose_token(logits, DecodeSettings(temperature=1.0, top_k=1, top_p=1.0), g) == 101
        assert choose_token(logits, DecodeSettings(temperature=1.0, top_k=100, top_p=0.1), g) == 101
    seen = {choose_token(logits, DecodeSettings(temperature=1.5, top_k=2, top_p=1.0), g) for _ in range(200)}
    assert seen == {101, 102}


def test_decoding_controls_affect_generated_output():
    m = _model()
    with torch.no_grad():  # sharpen the random model's logits so penalties have visible effect
        m.norm.weight.mul_(8.0)
    ids = build_prompt_ids(TOK, "sys", "hello")
    base = DecodeSettings(temperature=1.0, top_k=100, top_p=1.0, repetition_penalty=1.0, max_new_tokens=40, loop_guard=False, seed=1)
    out1, _ = generate(m, TOK, ids, base)
    out2, _ = generate(m, TOK, ids, DecodeSettings(**{**base.to_dict()}))
    assert out1 == out2  # seeded reproducibility
    variants = [
        {"temperature": 0.0}, {"top_k": 1}, {"top_p": 0.1}, {"repetition_penalty": 2.0}, {"seed": 2},
    ]
    for v in variants:
        out, _ = generate(m, TOK, ids, DecodeSettings(**{**base.to_dict(), **v}))
        assert out != out1, v
    short, _ = generate(m, TOK, ids, DecodeSettings(**{**base.to_dict(), "max_new_tokens": 5}))
    assert len(short) <= 5


def test_no_control_tokens_in_output_and_stop_conditions():
    m = _model(ctx=64)
    ids = build_prompt_ids(TOK, "s", "hi")
    events = list(stream_generate(m, TOK, ids, DecodeSettings(temperature=1.0, max_new_tokens=600, loop_guard=False, seed=0)))
    tokens = [e for e in events if e["type"] == "token"]
    assert all(e["id"] >= 6 for e in tokens)
    done = events[-1]
    assert done["type"] == "done" and done["stop_reason"] in ("eos", "context_full")
    if done["stop_reason"] == "context_full":
        assert len(ids) + len(tokens) == 64
    assert tokens[0]["ttft_ms"] > 0 and done["total_ms"] >= tokens[-1]["elapsed_ms"]


def test_cached_and_uncached_generation_agree():
    m = _model()
    ids = build_prompt_ids(TOK, "sys", "tell me")
    s = DecodeSettings(temperature=0.0, max_new_tokens=30, loop_guard=False, repetition_penalty=1.0)
    assert generate(m, TOK, ids, s, use_cache=True) == generate(m, TOK, ids, s, use_cache=False)


def test_fit_prompt_drops_oldest_exchanges():
    history = [("user", "u" * 60), ("assistant", "a" * 60), ("user", "short"), ("assistant", "ok")]
    ids, dropped = fit_prompt(TOK, "sys", "now", history, context_length=128, max_new_tokens=60)
    assert dropped == 2 and len(ids) <= 68
    with pytest.raises(ValueError):
        fit_prompt(TOK, "sys", "x" * 200, [], context_length=128, max_new_tokens=60)
