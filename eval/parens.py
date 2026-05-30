"""Parenthesis-matching probe (EXPERIMENTS.md §5.5).

Hierarchical-structure probe. Uses two reserved ids as OPEN and CLOSE. We
generate balanced bracket sequences up to depth `max_depth`, then measure how
often the model's next-token prediction respects the structure at CLOSE
positions:

    at each CLOSE token's prior position, does logits[OPEN] < logits[CLOSE]
    when the structure requires a CLOSE?

Reports accuracy of structurally-valid predictions across depths.
"""

from __future__ import annotations

import torch


def _reserved_ids(vocab_size: int) -> tuple[int, int]:
    if vocab_size < 8:
        raise ValueError("vocab_size too small for parens probe")
    return vocab_size - 5, vocab_size - 6  # OPEN, CLOSE


def _sample_balanced(
    *, seq_len: int, max_depth: int, open_id: int, close_id: int, generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy balanced-bracket sampler. Returns (seq, is_close_required) mask
    where the mask is True at positions where the *next* token must be CLOSE."""
    tokens: list[int] = []
    must_close: list[bool] = []
    depth = 0
    remaining_pairs = seq_len // 2
    for _ in range(seq_len):
        # Decide whether to open or close. Force close if at max depth or if
        # remaining slots equal current depth (must close to balance).
        slots_left = seq_len - len(tokens)
        force_close = depth >= max_depth or depth == slots_left
        force_open = depth == 0
        if force_open:
            choice = 0
        elif force_close:
            choice = 1
        else:
            # Slight bias toward open early, close late.
            p_open = max(0.3, 1.0 - len(tokens) / seq_len)
            choice = 0 if torch.rand((1,), generator=generator).item() < p_open else 1
        if choice == 0:
            tokens.append(open_id)
            must_close.append(False)
            depth += 1
        else:
            tokens.append(close_id)
            must_close.append(False)
            depth -= 1
        # If the *next* position must be a close (depth >= remaining), mark it.
        if depth > 0 and (seq_len - len(tokens)) == depth:
            if len(must_close) > 0:
                must_close[-1] = True
    return (
        torch.tensor(tokens, dtype=torch.long),
        torch.tensor(must_close, dtype=torch.bool),
    )


@torch.no_grad()
def run(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    device: torch.device,
    seq_len: int = 64,
    max_depth: int = 4,
    batch_size: int = 32,
    num_batches: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    model_max = int(getattr(model, "context_length", seq_len))
    seq_len = min(seq_len, model_max)
    if seq_len < 8:
        return {"capability/parens/acc": float("nan")}

    open_id, close_id = _reserved_ids(vocab_size)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()
    correct = 0
    total = 0
    try:
        for _ in range(num_batches):
            seqs, masks = [], []
            for _ in range(batch_size):
                s, m = _sample_balanced(
                    seq_len=seq_len, max_depth=max_depth,
                    open_id=open_id, close_id=close_id, generator=generator,
                )
                seqs.append(s)
                masks.append(m)
            seq = torch.stack(seqs, dim=0).to(device)
            mask = torch.stack(masks, dim=0).to(device)

            logits = model(seq).float()
            # mask is over current tokens; we care about predictions for the
            # *next* token, so logits[:, t, :] and mask[:, t].
            open_score = logits[..., open_id]
            close_score = logits[..., close_id]
            pred_close = close_score > open_score  # (B, T)
            # Score only at positions where the next token must be CLOSE.
            correct += int((pred_close & mask).sum().item())
            total += int(mask.sum().item())
    finally:
        if was_training:
            model.train()
    return {"capability/parens/acc": correct / max(total, 1)}
