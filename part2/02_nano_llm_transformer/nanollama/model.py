"""NanoLlama: a decoder-only transformer built from explicit primitives (F04).

Components: RMSNorm pre-normalization, interleaved rotary position embeddings on
queries/keys, causal multi-head self-attention with an optional key/value cache,
SwiGLU feed-forward blocks, and an LM head tied to the token embedding.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 104
    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 688
    context_length: int = 1024
    rope_base: float = 10000.0
    norm_eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")


def expected_param_count(cfg: ModelConfig) -> int:
    """Closed-form parameter count (embedding tied to the LM head)."""
    d, f = cfg.d_model, cfg.d_ff
    per_layer = 4 * d * d + 3 * d * f + 2 * d  # q,k,v,o + w1,w2,w3 + two RMSNorm gains
    return cfg.vocab_size * d + cfg.n_layers * per_layer + d  # + final norm


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * g — no mean subtraction, computed in float32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


def rope_tables(head_dim: int, max_positions: int, base: float = 10000.0):
    """cos/sin tables of shape [positions, head_dim] in the interleaved layout.

    Frequencies theta_i = base^(-2i/head_dim) are repeat-interleaved so coordinates
    (0,1), (2,3), ... share one rotation frequency.
    """
    inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    pos = torch.arange(max_positions, dtype=torch.float64)
    angles = torch.outer(pos, inv_freq).repeat_interleave(2, dim=-1)
    return angles.cos().float(), angles.sin().float()


def rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """(x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [T, D] already sliced at absolute positions."""
    return x * cos.to(x.dtype) + rotate_interleaved(x) * sin.to(x.dtype)


class KVCache:
    """Per-layer key/value tensors grown one step at a time during decoding."""

    def __init__(self, n_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers

    @property
    def length(self) -> int:
        k = self.keys[0]
        return 0 if k is None else k.shape[2]

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=2)
        return self.keys[layer], self.values[layer]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, layer_idx: int, cache: KVCache | None = None, return_attention: bool = False):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = None
        if cache is None and not return_attention:
            # Fused kernel path for full-sequence forward passes.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Explicit path: needed for the KV cache and for returning attention maps.
            if cache is not None:
                k, v = cache.update(layer_idx, k, v)
            S = k.shape[2]
            past = S - T
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # Upper-triangular -inf mask for the new tokens, left-padded with zeros
            # so each new token sees every cached position.
            mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
            if past:
                mask = torch.cat([torch.zeros(T, past, device=x.device), mask], dim=1)
            scores = scores + mask.to(scores.dtype)
            weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            out = weights @ v
            if return_attention:
                attn = weights
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out), attn


class SwiGLU(nn.Module):
    """W2( SiLU(W1 x) * W3 x ) with bias-free projections."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin, layer_idx, cache=None, return_attention=False):
        h, attn = self.attn(self.attn_norm(x), cos, sin, layer_idx, cache, return_attention)
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, attn


class NanoLlama(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        cos, sin = rope_tables(cfg.head_dim, cfg.context_length, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)
        # Scale residual output projections by depth (GPT-2 style) for stable training.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None, return_attention: bool = False):
        """Return (logits [B,T,V], attentions list[L] of [B,H,T,S] or None)."""
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if start + T > self.cfg.context_length:
            raise ValueError(f"sequence of {start + T} tokens exceeds context {self.cfg.context_length}")
        cos = self.rope_cos[start : start + T]
        sin = self.rope_sin[start : start + T]
        x = self.embed(idx)
        attentions = [] if return_attention else None
        for i, block in enumerate(self.blocks):
            x, attn = block(x, cos, sin, i, cache, return_attention)
            if return_attention:
                attentions.append(attn)
        logits = self.lm_head(self.norm(x))
        return logits, attentions

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.n_layers)
