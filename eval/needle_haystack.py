"""Needle-in-a-haystack probe (EXPERIMENTS.md §5.4).

A token-level approximation of the natural-language needle test. We construct
a sequence by:

    [REAL prefix from val tokens]  [DELIM]  [needle: K random tokens]
    [REAL middle from val tokens]  [QUERY DELIM]  [needle: K random tokens]

The needle is repeated verbatim at the end of the sequence. We measure
next-token accuracy on the second occurrence of the needle: does the model
recall what came right after the first `[DELIM]`?

Uses two reserved control tokens at the top of the vocab.
"""

from __future__ import annotations

import numpy as np
import torch


def _reserved_ids(vocab_size: int) -> tuple[int, int]:
    if vocab_size < 8:
        raise ValueError("vocab_size too small for needle probe")
    return vocab_size - 3, vocab_size - 4  # DELIM, QUERY_DELIM (distinct from selective_copy)


@torch.no_grad()
def run(
    model: torch.nn.Module,
    *,
    val_tokens: np.ndarray,
    vocab_size: int,
    device: torch.device,
    needle_k: int = 4,
    prefix_len: int = 16,
    middle_lens: tuple[int, ...] = (16, 64, 128),
    batch_size: int = 8,
    num_batches: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    model_max = int(getattr(model, "context_length", 1024))
    delim_id, query_id = _reserved_ids(vocab_size)
    ordinary = vocab_size - 4
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)

    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    try:
        for middle_len in middle_lens:
            total_seq_len = prefix_len + 1 + needle_k + middle_len + 1 + needle_k
            if total_seq_len > model_max or len(val_tokens) < prefix_len + middle_len + 8:
                results[f"capability/needle/acc@middle{middle_len}"] = float("nan")
                continue

            correct = 0
            total = 0
            for _ in range(num_batches):
                # Sample real prefix + real middle from disjoint regions of val.
                starts = rng.integers(0, len(val_tokens) - (prefix_len + middle_len) - 1, size=batch_size)
                prefix = np.stack([val_tokens[s : s + prefix_len] for s in starts])
                middle = np.stack([val_tokens[s + prefix_len : s + prefix_len + middle_len] for s in starts])

                prefix_t = torch.as_tensor(prefix, dtype=torch.long, device=device).clamp_max_(ordinary - 1)
                middle_t = torch.as_tensor(middle, dtype=torch.long, device=device).clamp_max_(ordinary - 1)
                needle = torch.randint(0, ordinary, (batch_size, needle_k), generator=generator, device=device)

                delim_col = torch.full((batch_size, 1), delim_id, dtype=torch.long, device=device)
                query_col = torch.full((batch_size, 1), query_id, dtype=torch.long, device=device)

                seq = torch.cat(
                    [prefix_t, delim_col, needle, middle_t, query_col, needle], dim=1
                )
                target_start = seq.shape[1] - needle_k

                logits = model(seq).float()
                pred = logits[:, target_start - 1 : target_start - 1 + needle_k, :].argmax(dim=-1)
                correct += int((pred == needle).sum().item())
                total += needle.numel()
            results[f"capability/needle/acc@middle{middle_len}"] = correct / max(total, 1)
    finally:
        if was_training:
            model.train()
    return results
