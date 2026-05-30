"""Reference baselines for capability metrics (EXPERIMENTS.md §6).

Provides "trivial" baselines so each capability score can be reported as a
delta against (a) random chance and (b) a bigram language model. The
Transformer-matched baseline lives in the sweep itself — these baselines are
the floor.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import torch


def random_baseline_acc(vocab_size: int) -> float:
    """Expected accuracy of a uniform random next-token predictor."""
    return 1.0 / max(int(vocab_size), 1)


def random_baseline_ppl(vocab_size: int) -> float:
    """Expected per-token perplexity of a uniform predictor."""
    return float(vocab_size)


def bigram_baseline_ppl(train_tokens: np.ndarray, val_tokens: np.ndarray, *, smoothing: float = 1.0) -> float:
    """Add-`smoothing` bigram LM. Returns per-token val perplexity.

    Cheap and pure-numpy; intended as a "minimum sensible" comparison.
    """
    if len(train_tokens) < 2 or len(val_tokens) < 2:
        return float("nan")

    vocab_size = int(max(train_tokens.max(), val_tokens.max())) + 1
    # Bigram counts over the train stream.
    counts: Counter[tuple[int, int]] = Counter()
    unigram: Counter[int] = Counter()
    for prev, cur in zip(train_tokens[:-1], train_tokens[1:]):
        counts[(int(prev), int(cur))] += 1
        unigram[int(prev)] += 1

    # Compute log-likelihood on val.
    log_lik = 0.0
    n = 0
    for prev, cur in zip(val_tokens[:-1], val_tokens[1:]):
        prev_i = int(prev)
        cur_i = int(cur)
        num = counts.get((prev_i, cur_i), 0) + smoothing
        denom = unigram.get(prev_i, 0) + smoothing * vocab_size
        log_lik += math.log(num / denom)
        n += 1
    avg_nll = -log_lik / max(n, 1)
    return math.exp(avg_nll)


@torch.no_grad()
def untrained_model_acc(
    build_fn,
    val_tokens: np.ndarray,
    *,
    context_length: int = 64,
    batch_size: int = 8,
    num_batches: int = 4,
    device: torch.device,
    seed: int = 0,
) -> float:
    """Greedy next-token accuracy of a freshly-initialised model (no training).

    `build_fn` is a no-arg callable returning an `nn.Module`. Used as an
    "architecture-only" floor — distinguishes the *effect of training* from
    the inductive biases of the architecture itself.
    """
    rng = np.random.default_rng(seed)
    model = build_fn().to(device)
    model.eval()
    correct = 0
    total = 0
    if len(val_tokens) < context_length + 2:
        return float("nan")
    for _ in range(num_batches):
        starts = rng.integers(0, len(val_tokens) - context_length - 1, size=batch_size)
        xs = np.stack([val_tokens[s : s + context_length] for s in starts])
        ys = np.stack([val_tokens[s + 1 : s + 1 + context_length] for s in starts])
        x = torch.as_tensor(xs, dtype=torch.long, device=device)
        y = torch.as_tensor(ys, dtype=torch.long, device=device)
        logits = model(x).float()
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.numel()
    return correct / max(total, 1)
