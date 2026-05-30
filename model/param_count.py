"""Closed-form parameter counters for matched-compute methodology.

The architecture-comparison sweep matches **active parameters** within ±5 %
across `transformer`, `ssm` (Mamba), and `subquadratic` (linear / sliding
attention). Computing these counts from a config (without instantiating the
model) lets the sweep script choose configs and verify the matching budget
before launching any training run.

Each function returns a `ParamBreakdown` (a typed dict-like dataclass) so the
caller can log a per-component view to W&B for debugging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Mapping

import torch


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class ParamBreakdown:
    """Per-component parameter count, plus the total."""

    arch: str
    embedding: int
    blocks: int           # sum across all layers
    final_norm: int
    lm_head: int
    total: int

    def as_dict(self) -> dict[str, int | str]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------


def _transformer_block_params(d_model: int, d_ff: int) -> int:
    """Per-block params for the orthodox Transformer in `model/transformer.py`.

    Linear layers in this codebase are weight-only (no bias).
    """
    norms = 2 * d_model                  # RMSNorm ln1 + ln2 (weight only)
    attn = 4 * d_model * d_model         # Q, K, V, output_proj
    ffn = 3 * d_model * d_ff             # SwiGLU w1, w2, w3
    return norms + attn + ffn


def transformer_params(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,  # unused here but kept in the signature for symmetry
    d_ff: int,
) -> ParamBreakdown:
    embedding = vocab_size * d_model
    blocks = num_layers * _transformer_block_params(d_model, d_ff)
    final_norm = d_model
    lm_head = d_model * vocab_size
    total = embedding + blocks + final_norm + lm_head
    return ParamBreakdown(
        arch="transformer",
        embedding=embedding,
        blocks=blocks,
        final_norm=final_norm,
        lm_head=lm_head,
        total=total,
    )


# ---------------------------------------------------------------------------
# SSM (Mamba)
# ---------------------------------------------------------------------------


def mamba_dt_rank(d_model: int) -> int:
    """Default Δ-projection rank from the Mamba paper: ceil(d_model / 16)."""
    return max(1, math.ceil(d_model / 16))


def _mamba_block_params(
    d_model: int, d_state: int, d_conv: int, expand: int
) -> int:
    """Per-block params for the Mamba block implemented in `model/ssm.py`."""
    d_inner = expand * d_model
    dt_rank = mamba_dt_rank(d_model)

    norm = d_model
    in_proj = d_model * (2 * d_inner)              # x and gate
    # Depthwise causal conv: weight shape (d_inner, 1, d_conv), bias d_inner
    conv = d_inner * d_conv + d_inner
    x_proj = d_inner * (dt_rank + 2 * d_state)     # produces Δ, B, C
    dt_proj = dt_rank * d_inner + d_inner          # with bias (Mamba inits bias)
    A_log = d_inner * d_state                       # learned log(-A)
    D_skip = d_inner                                # skip-connection scale
    out_proj = d_inner * d_model
    return norm + in_proj + conv + x_proj + dt_proj + A_log + D_skip + out_proj


def ssm_params(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_state: int,
    d_conv: int,
    expand: int,
) -> ParamBreakdown:
    embedding = vocab_size * d_model
    blocks = num_layers * _mamba_block_params(d_model, d_state, d_conv, expand)
    final_norm = d_model
    lm_head = d_model * vocab_size
    total = embedding + blocks + final_norm + lm_head
    return ParamBreakdown(
        arch="ssm",
        embedding=embedding,
        blocks=blocks,
        final_norm=final_norm,
        lm_head=lm_head,
        total=total,
    )


# ---------------------------------------------------------------------------
# Sub-quadratic attention
# ---------------------------------------------------------------------------


def subquadratic_params(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> ParamBreakdown:
    """Linear / sliding-window attention has the same parameter structure as the
    Transformer — only the attention operator changes, not the projections."""
    pb = transformer_params(vocab_size, d_model, num_layers, num_heads, d_ff)
    return ParamBreakdown(
        arch="subquadratic",
        embedding=pb.embedding,
        blocks=pb.blocks,
        final_norm=pb.final_norm,
        lm_head=pb.lm_head,
        total=pb.total,
    )


# ---------------------------------------------------------------------------
# Dispatch + verification
# ---------------------------------------------------------------------------


def count_active_params(arch: str, **cfg) -> ParamBreakdown:
    """Closed-form active-param count for any supported arch.

    Pass the same keyword args you would pass to `model.build_model`; unused
    keys are ignored.
    """
    arch = arch.lower()
    if arch == "transformer":
        return transformer_params(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
        )
    if arch == "ssm":
        return ssm_params(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            d_state=cfg.get("ssm_d_state", 16),
            d_conv=cfg.get("ssm_d_conv", 4),
            expand=cfg.get("ssm_expand", 2),
        )
    if arch == "subquadratic":
        return subquadratic_params(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            d_ff=cfg["d_ff"],
        )
    raise ValueError(f"Unknown arch '{arch}'")


def count_torch_params(model: torch.nn.Module) -> int:
    """Ground-truth parameter count for a constructed model (sanity check)."""
    return sum(p.numel() for p in model.parameters())


def within_tolerance(
    counts: Mapping[str, ParamBreakdown],
    reference: str = "transformer",
    tol: float = 0.05,
) -> dict[str, bool]:
    """Return per-arch boolean: is `total` within ±`tol` of the reference?"""
    if reference not in counts:
        raise KeyError(f"reference '{reference}' missing from counts")
    ref_total = counts[reference].total
    if ref_total <= 0:
        raise ValueError("reference total must be positive")
    out: dict[str, bool] = {}
    for arch, breakdown in counts.items():
        rel_err = abs(breakdown.total - ref_total) / ref_total
        out[arch] = rel_err <= tol
    return out
