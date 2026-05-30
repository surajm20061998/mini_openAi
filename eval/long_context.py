"""Long-context generalisation probe (EXPERIMENTS.md §5.1).

Train at `context_length=L_train`, evaluate val perplexity at progressively
longer eval contexts. Models that extrapolate well degrade less.

Reports `eval/ppl@L` for each L plus an extrapolation slope.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


@torch.no_grad()
def _val_loss_at_context(
    model: torch.nn.Module,
    val_tokens: np.ndarray,
    context_length: int,
    num_batches: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    total_tokens = len(val_tokens)
    if total_tokens < context_length + 1:
        return float("nan")
    starts_high = total_tokens - context_length - 1
    losses: list[float] = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(num_batches):
            starts = rng.integers(0, starts_high, size=batch_size)
            xs = np.stack([val_tokens[s : s + context_length] for s in starts])
            ys = np.stack([val_tokens[s + 1 : s + 1 + context_length] for s in starts])
            x = torch.as_tensor(xs, dtype=torch.long, device=device)
            y = torch.as_tensor(ys, dtype=torch.long, device=device)
            logits = model(x).float()
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
            losses.append(loss.item())
    finally:
        if was_training:
            model.train()
    return float(np.mean(losses))


def run(
    model: torch.nn.Module,
    *,
    val_tokens: np.ndarray,
    context_length_train: int,
    eval_lengths: Iterable[int],
    batch_size: int = 4,
    num_batches: int = 16,
    device: torch.device,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate val ppl at each `eval_lengths`. Caps each L at the model's
    `context_length` to avoid running past supported positions."""
    model_max = getattr(model, "context_length", context_length_train)
    results: dict[str, float] = {}
    for L in eval_lengths:
        L_use = min(int(L), int(model_max))
        loss = _val_loss_at_context(
            model, val_tokens, L_use, num_batches, batch_size, device, seed
        )
        ppl = math.exp(loss) if math.isfinite(loss) else float("nan")
        results[f"capability/long_ctx/loss@{L_use}"] = loss
        results[f"capability/long_ctx/ppl@{L_use}"] = ppl

    # Extrapolation slope: relative ppl change from train context to the
    # longest *supported* eval length.
    train_key = f"capability/long_ctx/ppl@{int(context_length_train)}"
    longest = max(int(min(L, model_max)) for L in eval_lengths)
    long_key = f"capability/long_ctx/ppl@{longest}"
    if train_key in results and long_key in results and results[train_key] > 0:
        results["capability/long_ctx/slope"] = (
            results[long_key] - results[train_key]
        ) / results[train_key]
    return results
