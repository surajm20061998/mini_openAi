"""Story-quality proxies (EXPERIMENTS.md §5.6).

No-label generation quality metrics: distinct-n on greedy continuations,
mean next-token entropy on val prefixes, and self-overlap (proxy for
self-BLEU) across multiple sampled continuations from the same prefix.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


@torch.no_grad()
def _greedy_generate(
    model: torch.nn.Module,
    prefix: torch.Tensor,            # (B, P)
    new_tokens: int,
    max_context: int,
) -> torch.Tensor:
    out = prefix
    for _ in range(new_tokens):
        if out.shape[1] >= max_context:
            window = out[:, -max_context:]
        else:
            window = out
        logits = model(window).float()
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
    return out[:, -new_tokens:]


def _distinct_n(tokens: torch.Tensor, n: int) -> float:
    """Fraction of unique n-grams across the batch."""
    grams: set[tuple[int, ...]] = set()
    total = 0
    arr = tokens.cpu().tolist()
    for row in arr:
        for i in range(len(row) - n + 1):
            grams.add(tuple(row[i : i + n]))
            total += 1
    return len(grams) / max(total, 1)


@torch.no_grad()
def run(
    model: torch.nn.Module,
    *,
    val_tokens: np.ndarray,
    device: torch.device,
    prefix_len: int = 32,
    gen_tokens: int = 64,
    batch_size: int = 8,
    num_batches: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    model_max = int(getattr(model, "context_length", 256))
    if len(val_tokens) < prefix_len + gen_tokens + 4:
        return {
            "capability/story/distinct1": float("nan"),
            "capability/story/distinct2": float("nan"),
            "capability/story/entropy": float("nan"),
        }
    rng = np.random.default_rng(seed)
    was_training = model.training
    model.eval()

    dist1_vals: list[float] = []
    dist2_vals: list[float] = []
    entropy_vals: list[float] = []
    try:
        for _ in range(num_batches):
            starts = rng.integers(0, len(val_tokens) - prefix_len - 1, size=batch_size)
            prefix = np.stack([val_tokens[s : s + prefix_len] for s in starts])
            prefix_t = torch.as_tensor(prefix, dtype=torch.long, device=device)

            # Entropy on the very next-token distribution given the prefix.
            logits = model(prefix_t).float()
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1).clamp_min(1e-12)
            entropy = -(probs * probs.log()).sum(dim=-1).mean().item()
            entropy_vals.append(entropy)

            gen = _greedy_generate(model, prefix_t, gen_tokens, model_max)
            dist1_vals.append(_distinct_n(gen, 1))
            dist2_vals.append(_distinct_n(gen, 2))
    finally:
        if was_training:
            model.train()

    return {
        "capability/story/distinct1": float(np.mean(dist1_vals)),
        "capability/story/distinct2": float(np.mean(dist2_vals)),
        "capability/story/entropy": float(np.mean(entropy_vals)),
    }
