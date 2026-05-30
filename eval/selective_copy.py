"""Selective-copy probe (EXPERIMENTS.md §5.2).

Synthetic task: copy K content tokens through `noise_len` distractors.

Sequence layout (left to right):
    [DELIM]  c_1 c_2 ... c_K  [DELIM]  d_1 d_2 ... d_N  [RECALL]  c_1 c_2 ... c_K

The metric is the model's next-token accuracy on the final `c_1..c_K` positions
— i.e., does it copy the early content tokens after seeing `[RECALL]`?

Three reserved token ids — `delim`, `recall`, and a `padding` floor for content
generation — are taken from the top of the vocabulary so they collide minimally
with frequent BPE tokens.
"""

from __future__ import annotations

import torch


def _reserved_ids(vocab_size: int) -> tuple[int, int]:
    """Pick two distinct ids near the top of the vocab as control tokens."""
    if vocab_size < 8:
        raise ValueError("vocab_size too small for selective-copy probe")
    return vocab_size - 1, vocab_size - 2  # DELIM, RECALL


def _build_batch(
    *,
    batch_size: int,
    content_k: int,
    noise_len: int,
    vocab_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    delim_id, recall_id = _reserved_ids(vocab_size)
    # Content / noise drawn from "ordinary" ids, leaving the top two free.
    ordinary = vocab_size - 2
    content = torch.randint(0, ordinary, (batch_size, content_k), generator=generator, device=device)
    noise = torch.randint(0, ordinary, (batch_size, noise_len), generator=generator, device=device)

    delim_col = torch.full((batch_size, 1), delim_id, dtype=torch.long, device=device)
    recall_col = torch.full((batch_size, 1), recall_id, dtype=torch.long, device=device)

    seq = torch.cat([delim_col, content, delim_col, noise, recall_col, content], dim=1)

    # Target positions: the last `content_k` tokens of the sequence.
    target_start = seq.shape[1] - content_k
    return seq, content, target_start


@torch.no_grad()
def run(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    device: torch.device,
    content_k: int = 5,
    distances: tuple[int, ...] = (8, 32, 128),
    batch_size: int = 16,
    num_batches: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Run the probe at each distractor `distances` value."""
    model_max = int(getattr(model, "context_length", 1024))
    generator = torch.Generator(device=device).manual_seed(seed)

    was_training = model.training
    model.eval()
    results: dict[str, float] = {}
    try:
        for noise_len in distances:
            total_seq_len = 1 + content_k + 1 + noise_len + 1 + content_k
            if total_seq_len > model_max:
                results[f"capability/selective_copy/acc@{noise_len}"] = float("nan")
                continue

            correct = 0
            total = 0
            for _ in range(num_batches):
                seq, content, target_start = _build_batch(
                    batch_size=batch_size,
                    content_k=content_k,
                    noise_len=noise_len,
                    vocab_size=vocab_size,
                    device=device,
                    generator=generator,
                )
                logits = model(seq).float()
                # At position `t`, logits predict token `t+1`. So to predict
                # the content tokens that live at `[target_start, target_start+K)`,
                # we read logits at `[target_start-1, target_start+K-1)`.
                pred = logits[:, target_start - 1 : target_start - 1 + content_k, :].argmax(dim=-1)
                correct += int((pred == content).sum().item())
                total += content.numel()
            results[f"capability/selective_copy/acc@{noise_len}"] = correct / max(total, 1)
    finally:
        if was_training:
            model.train()
    return results
