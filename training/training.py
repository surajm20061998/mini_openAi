from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional, IO, BinaryIO

import numpy as np
import torch
from torch import Tensor


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Numerically stable cross entropy over arbitrary leading batch dims.

    logits: (..., vocab_size)  (unnormalized)
    targets: (...)  (int64 indices)
    returns: scalar mean loss
    """
    if logits.ndim < 1:
        raise ValueError("logits must have at least 1 dimension")
    if targets.shape != logits.shape[:-1]:
        raise ValueError(f"targets shape {targets.shape} must match logits batch shape {logits.shape[:-1]}")

    x = logits.to(torch.float32)

    x_max = torch.amax(x, dim=-1, keepdim=True)
    x_shift = x - x_max 

    lse = torch.log(torch.sum(torch.exp(x_shift), dim=-1))

    tgt = targets.to(torch.long).unsqueeze(-1)
    x_tgt = torch.gather(x_shift, dim=-1, index=tgt).squeeze(-1)

    loss = lse - x_tgt

    return loss.mean().to(logits.dtype)


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    """
    dataset: 1D numpy array of token ids
    returns:
      x: (batch_size, context_length)
      y: (batch_size, context_length)  (next-token labels)
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D numpy array of token ids")
    n = int(dataset.shape[0])
    if n <= context_length:
        raise ValueError("dataset too small for given context_length")

    dev = torch.device(device)
    data = torch.as_tensor(dataset, dtype=torch.long, device=dev)

    max_start = n - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,), device=dev)
    offsets = torch.arange(context_length, device=dev) 

    idx = starts[:, None] + offsets[None, :] 
    x = data[idx]
    y = data[idx + 1]
    return x, y


def clip_grad_l2(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """
    In-place gradient clipping by global L2 norm.
    """
    if max_l2_norm <= 0:
        return

    total_sq = torch.tensor(0.0, dtype=torch.float32)
    grads = []
    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad.data
        grads.append(g)
        total_sq = total_sq + torch.sum(g.to(torch.float32) * g.to(torch.float32))

    total_norm = torch.sqrt(total_sq)
    if torch.isnan(total_norm) or torch.isinf(total_norm):
        return

    if total_norm > max_l2_norm:
        scale = float(max_l2_norm) / float(total_norm + eps)
        for g in grads:
            g.mul_(scale)


class AdamW(torch.optim.Optimizer):
    """
    AdamW as in Loshchilov & Hutter (2019), decoupled weight decay.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0:
            raise ValueError(f"Invalid eps: {eps}")
        b1, b2 = betas
        if not (0.0 <= b1 < 1.0 and 0.0 <= b2 < 1.0):
            raise ValueError(f"Invalid betas: {betas}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            b1, b2 = group["betas"]
            eps = float(group["eps"])
            wd = float(group["weight_decay"])

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                t = state.get("t", 0) + 1 

                if "m" not in state:
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m = state["m"]
                v = state["v"]

                m.mul_(b1).add_(grad, alpha=(1.0 - b1))
                v.mul_(b2).addcmul_(grad, grad, value=(1.0 - b2))

                bias_c1 = 1.0 - (b1 ** t)
                bias_c2 = 1.0 - (b2 ** t)
                step_size = lr * math.sqrt(bias_c2) / bias_c1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size)

                if wd != 0.0:
                    p.data.add_(p.data, alpha=-lr * wd)

                state["t"] = t

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    LLaMA-style: linear warmup then cosine decay then flat.
    """
    t = int(it)
    Tw = int(warmup_iters)
    Tc = int(cosine_cycle_iters)

    if t < 0:
        t = 0

    if Tw > 0 and t < Tw:
        return (t / Tw) * max_learning_rate

    if t <= Tc:
        if Tc == Tw:
            return min_learning_rate
        progress = (t - Tw) / (Tc - Tw) 
        return min_learning_rate + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_learning_rate - min_learning_rate)

    return min_learning_rate


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | BinaryIO | IO[bytes],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(payload, out)


def load_checkpoint(
    src: str | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    payload = torch.load(src, map_location="cpu")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    return int(payload["iteration"])