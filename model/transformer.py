from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    x_max = torch.amax(x, dim=dim, keepdim=True)
    exp = torch.exp(x - x_max)
    return exp / torch.sum(exp, dim=dim, keepdim=True)

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:

        return torch.einsum("... i, o i -> ... o", x, self.weight)

class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )
        # N(0,1) truncated at [-3, 3]
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x_f = x.to(torch.float32)
        # RMS(a) = sqrt(mean(a^2) + eps)
        rms = torch.sqrt(torch.mean(x_f * x_f, dim=-1, keepdim=True) + self.eps)
        y = (x_f / rms) * self.weight.to(torch.float32)
        return y.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires even d_k"
        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        inv_freq = self.theta ** (
            -torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k
        )
      
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.einsum("p, j -> p j", positions, inv_freq) 

        cos = torch.cos(angles)  
        sin = torch.sin(angles)  

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len) (same leading batch dims as x without the last d_k)
        """
        cos = self.cos[token_positions] 
        sin = self.sin[token_positions]  

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """
    Q: (..., q, d_k)
    K: (..., k, d_k)
    V: (..., k, d_v)
    mask: (..., q, k) boolean where True means "allowed"
    """
    d_k = Q.shape[-1]
    scores = torch.einsum("... q d, ... k d -> ... q k", Q.to(torch.float32), K.to(torch.float32))
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        neg_inf = torch.tensor(-float("inf"), device=scores.device, dtype=scores.dtype)
        scores = torch.where(mask, scores, neg_inf)

    attn = softmax(scores, dim=-1)
    out = torch.einsum("... q k, ... k d -> ... q d", attn, V.to(torch.float32))
    return out.to(Q.dtype)

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rope: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.use_rope = use_rope
        if use_rope:
            assert max_seq_len is not None and theta is not None
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)
        else:
            self.rope = None

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        """
        x: (..., seq, d_model)
        token_positions: (..., seq) optional
        """
        *batch_dims, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        k = k.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        v = v.view(*batch_dims, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)

        if self.use_rope:
            if token_positions is None:
                pos = torch.arange(seq_len, device=x.device, dtype=torch.long)
                token_positions = pos.view(*([1] * len(batch_dims)), seq_len).expand(*batch_dims, seq_len)
 
            q = self.rope(q, token_positions.unsqueeze(-2).expand(*batch_dims, self.num_heads, seq_len))
            k = self.rope(k, token_positions.unsqueeze(-2).expand(*batch_dims, self.num_heads, seq_len))

        causal = torch.tril(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool))
        causal = causal.view(*([1] * (len(batch_dims) + 1)), seq_len, seq_len).expand(*batch_dims, self.num_heads, seq_len, seq_len)

        out = scaled_dot_product_attention(q, k, v, mask=causal)

        out = out.transpose(-3, -2).contiguous().view(*batch_dims, seq_len, self.d_model)
        return self.output_proj(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=True,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
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
            raise ValueError(f"sequence_length {seq} exceeds context_length {self.context_length}")

        x = self.token_embeddings(in_indices)

        token_positions = torch.arange(seq, device=in_indices.device, dtype=torch.long).view(1, seq).expand(batch, seq)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits