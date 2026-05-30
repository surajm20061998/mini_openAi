"""Induction-head probe (EXPERIMENTS.md §5.3).

Plants a bigram `A B` early in a random sequence, then `A` again later, and
checks whether the model predicts `B` after the second `A`. This is the
quintessential demonstration of in-context bigram completion: Transformers
form induction heads readily; SSMs/sub-quadratic mixers vary.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def run(
    model: torch.nn.Module,
    *,
    vocab_size: int,
    device: torch.device,
    seq_len: int = 64,
    batch_size: int = 32,
    num_batches: int = 8,
    seed: int = 0,
) -> dict[str, float]:
    """Returns the second-occurrence prediction accuracy."""
    model_max = int(getattr(model, "context_length", seq_len))
    seq_len = min(seq_len, model_max)
    if seq_len < 8:
        return {"capability/induction/acc": float("nan")}

    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()

    correct = 0
    total = 0
    try:
        for batch_idx in range(num_batches):
            seq = torch.randint(
                0, vocab_size, (batch_size, seq_len),
                generator=generator, device=device,
            )

            # Random A, B per row, distinct from each other.
            a = torch.randint(0, vocab_size, (batch_size,), generator=generator, device=device)
            b = torch.randint(0, vocab_size, (batch_size,), generator=generator, device=device)
            # Ensure A != B; on collision, bump B by 1 modulo vocab.
            mask = (a == b)
            b = torch.where(mask, (b + 1) % vocab_size, b)

            # Pick planting positions: one near the start, one near the end
            # (with enough gap to test "long-range" induction).
            pos1 = torch.randint(1, seq_len // 4, (batch_size,), generator=generator, device=device)
            pos2 = torch.randint(seq_len // 2, seq_len - 2, (batch_size,), generator=generator, device=device)

            row = torch.arange(batch_size, device=device)
            # Plant AB at pos1, A at pos2; B should be predicted at pos2.
            seq[row, pos1] = a
            seq[row, pos1 + 1] = b
            seq[row, pos2] = a

            logits = model(seq).float()
            # logits[:, t, :] predicts token at t+1; we want prediction for pos2+1.
            pred = logits[row, pos2, :].argmax(dim=-1)
            correct += int((pred == b).sum().item())
            total += batch_size
    finally:
        if was_training:
            model.train()
    return {"capability/induction/acc": correct / max(total, 1)}
