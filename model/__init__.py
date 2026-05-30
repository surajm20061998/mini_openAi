"""Model architecture registry.

Adds a thin `build_model` factory so the trainer can dispatch on a
`--model_arch` string instead of importing a specific class. Existing
Transformer behavior is unchanged — passing `arch="transformer"` is
identical to instantiating `TransformerLM` directly.

State-space and sub-quadratic backends are wired here but stubbed in
their respective modules; see EXPERIMENTS.md for the roadmap.
"""

from __future__ import annotations

from typing import Optional

import torch

from .transformer import TransformerLM

__all__ = ["TransformerLM", "build_model", "SUPPORTED_ARCHS"]

SUPPORTED_ARCHS = ("transformer", "ssm", "subquadratic")


def build_model(
    arch: str,
    *,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    # Extra knobs reserved for non-Transformer backends. They are accepted
    # here so the trainer can pass them unconditionally; each backend pulls
    # what it needs and ignores the rest.
    ssm_d_state: int = 16,
    ssm_d_conv: int = 4,
    ssm_expand: int = 2,
    subq_kind: str = "linear",
    subq_window_size: int = 128,
    subq_feature_map: str = "elu",
) -> torch.nn.Module:
    """Instantiate a language model by architecture name.

    Parameters mirror `TransformerLM` for the shared knobs (vocab, depth,
    width, heads, ff, rope). Backend-specific knobs are passed by keyword
    and only consumed by the relevant backend.
    """
    arch = arch.lower()
    if arch == "transformer":
        return TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            device=device,
            dtype=dtype,
        )
    if arch == "ssm":
        # Imported lazily so a missing/incomplete backend never breaks the
        # default Transformer training path.
        from .ssm import SSMLM

        return SSMLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            d_state=ssm_d_state,
            d_conv=ssm_d_conv,
            expand=ssm_expand,
            device=device,
            dtype=dtype,
        )
    if arch == "subquadratic":
        from .subquadratic import SubquadraticLM

        return SubquadraticLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            attention_kind=subq_kind,
            window_size=subq_window_size,
            feature_map=subq_feature_map,
            device=device,
            dtype=dtype,
        )
    raise ValueError(
        f"Unknown model arch '{arch}'. Supported: {SUPPORTED_ARCHS}"
    )
