"""State-space language model — Mamba (selective S6).

A reference-quality implementation of the Mamba block from "Mamba: Linear-Time
Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023). The selective
scan is implemented as an explicit Python loop over the time dimension: this is
slow per token but numerically faithful and works without a CUDA kernel
(important for MPS / CPU). Optimised kernels can replace `_selective_scan`
later without changing the surrounding code.

Block structure (per layer):

    x ──RMSNorm──▶ in_proj ──▶ split ────────────▶ x_branch ─┐
                                  │                          │
                                  └──▶ z_gate ────▶ SiLU ────┤
                                                             ▼
        x_branch ─▶ Conv1d_causal ─▶ SiLU ─▶ x_proj ─▶ (Δ,B,C)
                                                │
                                          (A,D learned)
                                                ▼
                                       _selective_scan
                                                │
                                                ▼
                                          y = scan + D·x
                                          y = y * silu(z)
                                          y ─▶ out_proj ─▶ residual

Embedding → N × MambaBlock → RMSNorm → lm_head, matching the
`TransformerLM` interface so the trainer treats them interchangeably.

See EXPERIMENTS.md §2.1 for the SSM track plan.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .transformer import Embedding, Linear, RMSNorm, silu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mamba_dt_rank(d_model: int) -> int:
    """Default Δ-projection rank from the Mamba paper: ceil(d_model / 16)."""
    return max(1, math.ceil(d_model / 16))


def _selective_scan(
    x: Tensor,        # (B, L, d_inner)
    delta: Tensor,    # (B, L, d_inner)
    A: Tensor,        # (d_inner, d_state)
    B: Tensor,        # (B, L, d_state)
    C: Tensor,        # (B, L, d_state)
) -> Tensor:
    """Sequential selective scan in float32.

    Implements, per time step `t`:
        h[t] = exp(Δ[t] · A) · h[t-1] + (Δ[t] · B[t]) · x[t]
        y[t] = sum_j C[t,j] · h[t,:,j]

    Shapes:
        h : (B, d_inner, d_state)
        y : (B, L, d_inner)
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[1]
    device = x.device

    # Promote to float32 inside the recurrence for numerical stability — the
    # exponential of Δ·A can blow up in low precision.
    x_f = x.to(torch.float32)
    delta_f = delta.to(torch.float32)
    A_f = A.to(torch.float32)
    B_f = B.to(torch.float32)
    C_f = C.to(torch.float32)

    h = torch.zeros(batch, d_inner, d_state, device=device, dtype=torch.float32)
    ys: list[Tensor] = []

    # Broadcasting in the loop:
    #   delta_f[:, t, :]  -> (B, d_inner)        gives (B, d_inner, 1)
    #   A_f               -> (d_inner, d_state)  broadcasts to (1, d_inner, d_state)
    #   B_f[:, t, :]      -> (B, d_state)        gives (B, 1, d_state)
    #   x_f[:, t, :]      -> (B, d_inner)        gives (B, d_inner, 1)
    #   C_f[:, t, :]      -> (B, d_state)        gives (B, 1, d_state)
    A_b = A_f.unsqueeze(0)  # (1, d_inner, d_state)
    for t in range(seq_len):
        d_t = delta_f[:, t, :].unsqueeze(-1)      # (B, d_inner, 1)
        dA = torch.exp(d_t * A_b)                  # (B, d_inner, d_state)
        dB = d_t * B_f[:, t, :].unsqueeze(1)       # (B, d_inner, d_state)
        h = dA * h + dB * x_f[:, t, :].unsqueeze(-1)
        y_t = (h * C_f[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        ys.append(y_t)

    y = torch.stack(ys, dim=1)  # (B, L, d_inner)
    return y.to(x.dtype)


# ---------------------------------------------------------------------------
# Mamba block
# ---------------------------------------------------------------------------


class MambaBlock(nn.Module):
    """Single Mamba (selective S6) block with pre-norm residual."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.d_inner = expand * d_model
        self.dt_rank = _mamba_dt_rank(d_model)

        self.norm = RMSNorm(d_model, device=device, dtype=dtype)

        # Input projection: produces (x_branch, z_gate)
        self.in_proj = Linear(d_model, 2 * self.d_inner, device=device, dtype=dtype)

        # Depthwise causal Conv1d. Padding `d_conv - 1` then trim — keeps it
        # strictly causal regardless of kernel size.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
            device=device,
            dtype=dtype,
        )

        # Selective projections: x_inner -> (Δ_rank, B, C)
        self.x_proj = Linear(
            self.d_inner, self.dt_rank + 2 * d_state, device=device, dtype=dtype
        )
        # Δ projection: Δ_rank -> d_inner (with bias, as in the Mamba paper).
        # We use nn.Linear here because we want the bias term (the custom
        # Linear in this codebase is weight-only).
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, device=device, dtype=dtype
        )

        # A is parameterised as log(-A) for unconstrained optimisation; the
        # actual recurrence uses A = -exp(A_log) which is guaranteed negative.
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device)
        A = A.unsqueeze(0).expand(self.d_inner, d_state).contiguous()
        self.A_log = nn.Parameter(torch.log(A).to(dtype=dtype) if dtype else torch.log(A))

        # D is a learnable per-channel skip connection.
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device, dtype=dtype))

        # Output projection back to d_model.
        self.out_proj = Linear(self.d_inner, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)."""
        residual = x
        x_norm = self.norm(x)
        batch, seq_len, _ = x_norm.shape

        # (B, L, 2*d_inner) -> (B, L, d_inner), (B, L, d_inner)
        x_and_z = self.in_proj(x_norm)
        x_branch, z_gate = x_and_z.chunk(2, dim=-1)

        # Causal depthwise conv: rearrange to (B, d_inner, L), apply, trim.
        x_branch_t = x_branch.transpose(1, 2)                        # (B, d_inner, L)
        x_conv = self.conv1d(x_branch_t)[:, :, :seq_len]             # causal trim
        x_branch = x_conv.transpose(1, 2)                            # (B, L, d_inner)
        x_branch = silu(x_branch)

        # Selective Δ, B, C projection.
        x_dbl = self.x_proj(x_branch)                                # (B, L, dt_rank + 2*d_state)
        dt_part, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = self.dt_proj(dt_part)                                # (B, L, d_inner)
        # softplus keeps Δ positive (Mamba uses softplus for the step size).
        delta = nn.functional.softplus(delta)

        # Recover A and run the scan.
        A = -torch.exp(self.A_log)                                   # (d_inner, d_state)
        y_scan = _selective_scan(x_branch, delta, A, B_param, C_param)

        # D-skip + gate + project out.
        y = y_scan + self.D * x_branch
        y = y * silu(z_gate)
        out = self.out_proj(y)

        return residual + out


# ---------------------------------------------------------------------------
# Language model
# ---------------------------------------------------------------------------


class SSMLM(nn.Module):
    """Decoder-only LM with Mamba sequence mixing.

    Same forward signature as `TransformerLM` so the trainer doesn't care which
    backend it's running.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand

        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
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
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)
