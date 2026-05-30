"""Sub-quadratic attention language model.

Provides two attention variants under the same `SubquadraticLM` shell so the
trainer and sweep treat them as one architecture family:

* `attention_kind="linear"` — kernel-feature-map linear attention
  (Katharopoulos et al., 2020). Causality is enforced with a cumulative-sum
  formulation that is O(L · d_model · d_head). The feature map is
  `elu(x)+1` (default) or `relu(x)`.

* `attention_kind="sliding"` — local windowed softmax attention with radius
  `window_size`. Compute is O(L · w · d_model). Useful as a second variant
  to disentangle "sub-quadratic in L" from "kernel approximation to softmax".

Everything else (token embedding, RMSNorm, RoPE, SwiGLU FFN, pre-norm
residuals, LM head) is reused from `model.transformer` so the parameter
structure exactly matches the Transformer baseline at the same
`d_model`/`num_heads`/`d_ff` — which is precisely what we want for the
matched-active-params comparison.

See EXPERIMENTS.md §2.2 for the sub-quadratic track plan.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .transformer import (
    Embedding,
    Linear,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    softmax,
)


# ---------------------------------------------------------------------------
# Feature maps for linear attention
# ---------------------------------------------------------------------------


def _feature_map_elu(x: Tensor) -> Tensor:
    return nn.functional.elu(x) + 1.0


def _feature_map_relu(x: Tensor) -> Tensor:
    return nn.functional.relu(x)


def _get_feature_map(name: str):
    name = name.lower()
    if name == "elu":
        return _feature_map_elu
    if name == "relu":
        return _feature_map_relu
    raise ValueError(f"Unknown feature_map '{name}'. Use 'elu' or 'relu'.")


# ---------------------------------------------------------------------------
# Linear attention (causal, cumulative-sum formulation)
# ---------------------------------------------------------------------------


def _linear_causal_attention(
    q: Tensor,  # (B, H, L, d_head)
    k: Tensor,  # (B, H, L, d_head)
    v: Tensor,  # (B, H, L, d_head)
    feature_map,
    eps: float = 1e-6,
) -> Tensor:
    """Causal linear attention via cumulative sums.

    Numerator   : φ(Q_t) · Σ_{s≤t} φ(K_s) ⊗ V_s
    Denominator : φ(Q_t) · Σ_{s≤t} φ(K_s)
    Output      : numerator / (denominator + eps)
    """
    phi_q = feature_map(q)
    phi_k = feature_map(k)

    # Outer product φ(K) ⊗ V per token, then cumulative sum along L.
    # Shape: (B, H, L, d_head, d_head)
    kv = torch.einsum("bhld,bhle->bhlde", phi_k, v)
    kv_cum = kv.cumsum(dim=2)

    # Cumulative sum of φ(K) for the denominator. (B, H, L, d_head)
    k_cum = phi_k.cumsum(dim=2)

    # Numerator: project φ(Q) against the cumulative kv. (B, H, L, d_head)
    num = torch.einsum("bhld,bhlde->bhle", phi_q, kv_cum)

    # Denominator: φ(Q) · cumulative φ(K), summed over feature dim.
    denom = torch.einsum("bhld,bhld->bhl", phi_q, k_cum).unsqueeze(-1)

    return num / (denom + eps)


# ---------------------------------------------------------------------------
# Sliding-window attention (causal, local softmax)
# ---------------------------------------------------------------------------


def _sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window_size: int,
) -> Tensor:
    """Standard scaled-dot-product attention restricted to a local window.

    Mask: `(j <= i) and (i - j < window_size)`. Compute is O(L · w · d_head).
    """
    d_head = q.shape[-1]
    seq_len = q.shape[-2]
    device = q.device

    scores = torch.einsum("b h q d, b h k d -> b h q k", q.to(torch.float32), k.to(torch.float32))
    scores = scores / math.sqrt(d_head)

    positions = torch.arange(seq_len, device=device)
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L), diff[i,j] = i - j
    mask = (diff >= 0) & (diff < window_size)               # (L, L) bool
    mask = mask.view(1, 1, seq_len, seq_len)

    neg_inf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
    scores = torch.where(mask, scores, neg_inf)

    attn = softmax(scores, dim=-1)
    out = torch.einsum("b h q k, b h k d -> b h q d", attn, v.to(torch.float32))
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Block + LM
# ---------------------------------------------------------------------------


class SubquadraticAttention(nn.Module):
    """Multi-head attention layer with a pluggable sub-quadratic kernel.

    Mirrors `MultiHeadSelfAttention` in `model.transformer` — same Q/K/V/output
    projection structure, optional RoPE on Q and K — but routes through
    `_linear_causal_attention` or `_sliding_window_attention` depending on
    `kind`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float,
        kind: str = "linear",
        window_size: int = 128,
        feature_map: str = "elu",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "num_heads must divide d_model"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kind = kind.lower()
        self.window_size = int(window_size)
        self.feature_map_name = feature_map.lower()
        self._feature_map = _get_feature_map(feature_map)

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        # Share RoPE convention with the Transformer baseline so positional
        # information is fairly matched.
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device
        )

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if token_positions is None:
            token_positions = (
                torch.arange(seq_len, device=x.device, dtype=torch.long)
                .view(1, seq_len)
                .expand(batch, seq_len)
            )
        # RoPE expects (..., seq, d_head); broadcast positions across heads.
        positions_heads = token_positions.unsqueeze(1).expand(batch, self.num_heads, seq_len)
        q = self.rope(q, positions_heads)
        k = self.rope(k, positions_heads)

        if self.kind == "linear":
            out = _linear_causal_attention(q, k, v, feature_map=self._feature_map)
        elif self.kind == "sliding":
            out = _sliding_window_attention(q, k, v, window_size=self.window_size)
        else:
            raise ValueError(
                f"Unknown attention_kind '{self.kind}'. Use 'linear' or 'sliding'."
            )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.output_proj(out)


class SubquadraticBlock(nn.Module):
    """Pre-norm residual block — RMSNorm → SubquadraticAttention → RMSNorm → SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        kind: str,
        window_size: int,
        feature_map: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = SubquadraticAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            kind=kind,
            window_size=window_size,
            feature_map=feature_map,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class SubquadraticLM(nn.Module):
    """Decoder-only LM with sub-quadratic sequence mixing."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        attention_kind: str = "linear",
        window_size: int = 128,
        feature_map: str = "elu",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.attention_kind = attention_kind.lower()
        self.window_size = int(window_size)
        self.feature_map = feature_map.lower()

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                SubquadraticBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    kind=self.attention_kind,
                    window_size=self.window_size,
                    feature_map=self.feature_map,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        batch, seq = in_indices.shape
        if seq > self.context_length:
            raise ValueError(
                f"sequence_length {seq} exceeds context_length {self.context_length}"
            )
        x = self.token_embeddings(in_indices)
        token_positions = (
            torch.arange(seq, device=in_indices.device, dtype=torch.long)
            .view(1, seq)
            .expand(batch, seq)
        )
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        x = self.ln_final(x)
        return self.lm_head(x)
