import torch
import torch.nn.functional as F

from nanollama.model import (NanoLlama, RMSNorm, SwiGLU, apply_rope, expected_param_count, rope_tables)
from tests.conftest import tiny_model_config


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    return NanoLlama(tiny_model_config(**kw)).eval()


def test_rmsnorm_matches_formula_without_mean_subtraction():
    norm = RMSNorm(6, eps=1e-5)
    with torch.no_grad():
        norm.weight.copy_(torch.arange(1.0, 7.0))
    x = torch.randn(3, 6) + 5.0
    expected = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + 1e-5) * norm.weight
    assert torch.allclose(norm(x), expected, atol=1e-6)


def test_rope_rotates_adjacent_pairs_and_depends_on_relative_offset():
    d = 8
    cos, sin = rope_tables(d, 64)
    x = torch.randn(1, 1, 1, d)
    pos = 5
    out = apply_rope(x, cos[pos : pos + 1], sin[pos : pos + 1])
    manual = x.clone()
    for i in range(d // 2):
        theta = pos * 10000 ** (-2 * i / d)
        a, b = x[..., 2 * i], x[..., 2 * i + 1]
        manual[..., 2 * i] = a * torch.cos(torch.tensor(theta)) - b * torch.sin(torch.tensor(theta))
        manual[..., 2 * i + 1] = a * torch.sin(torch.tensor(theta)) + b * torch.cos(torch.tensor(theta))
    assert torch.allclose(out, manual, atol=1e-5)

    q, k = torch.randn(1, 1, 1, d), torch.randn(1, 1, 1, d)
    def dot(pq, pk):
        return float((apply_rope(q, cos[pq:pq + 1], sin[pq:pq + 1]) * apply_rope(k, cos[pk:pk + 1], sin[pk:pk + 1])).sum())
    assert abs(dot(10, 7) - dot(40, 37)) < 1e-4
    assert abs(dot(10, 7) - dot(10, 2)) > 1e-4


def test_swiglu_formula():
    cfg = tiny_model_config()
    ff = SwiGLU(cfg)
    x = torch.randn(2, 3, cfg.d_model)
    assert torch.allclose(ff(x), ff.w2(F.silu(ff.w1(x)) * ff.w3(x)))
    assert ff.w1.bias is None and ff.w2.bias is None and ff.w3.bias is None


def test_weight_tying_and_parameter_count():
    m = _model()
    assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr()
    assert m.num_parameters() == expected_param_count(m.cfg)
    from nanollama.model import ModelConfig
    assert expected_param_count(ModelConfig()) == NanoLlama(ModelConfig()).num_parameters() == 4_773_120


def test_causal_future_tokens_do_not_change_past_logits():
    m = _model()
    a = torch.randint(6, 104, (1, 40))
    b = a.clone()
    b[0, 25:] = torch.randint(6, 104, (15,))
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[0, :25], lb[0, :25], atol=1e-5)
    assert not torch.allclose(la[0, 25:], lb[0, 25:])


def test_fused_and_explicit_attention_paths_agree():
    m = _model()
    x = torch.randint(6, 104, (2, 50))
    with torch.no_grad():
        fused, none = m(x)
        explicit, attn = m(x, return_attention=True)
    assert none is None
    assert torch.allclose(fused, explicit, atol=1e-4)
    assert len(attn) == m.cfg.n_layers and attn[0].shape == (2, m.cfg.n_heads, 50, 50)
    assert torch.allclose(attn[1].sum(-1), torch.ones(2, m.cfg.n_heads, 50), atol=1e-5)
    upper = torch.triu(torch.ones(50, 50, dtype=torch.bool), 1)
    assert attn[0][..., upper].abs().max() == 0


def test_kv_cache_incremental_logits_match_full_forward():
    m = _model()
    x = torch.randint(6, 104, (1, 30))
    with torch.no_grad():
        full, _ = m(x)
        cache = m.new_cache()
        prefill, _ = m(x[:, :12], cache=cache)
        steps = [prefill[:, -1]]
        for t in range(12, 30):
            out, _ = m(x[:, t : t + 1], cache=cache)
            steps.append(out[:, -1])
    assert cache.length == 30
    incremental = torch.stack(steps[:-1], dim=1)  # predictions for positions 11..28
    assert torch.allclose(full[:, 11:29], incremental, atol=1e-4)
    assert torch.allclose(full[:, 29], steps[-1], atol=1e-4)


def test_context_limit_is_enforced():
    m = _model(context_length=16)
    import pytest
    with pytest.raises(ValueError):
        m(torch.zeros(1, 17, dtype=torch.long))
