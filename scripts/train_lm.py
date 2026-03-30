from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import multiprocessing as mp

import numpy as np
import torch

from model.transformer import TransformerLM
from training.training import (
    AdamW,
    cross_entropy_loss,
    clip_grad_l2,
    get_lr_cosine_schedule,
    save_checkpoint,
    load_checkpoint,
)

def _batch_worker(
    tokens_path: str,
    batch_size: int,
    context_length: int,
    seed: int,
    out_q: "mp.Queue[Tuple[np.ndarray, np.ndarray]]",
    stop_ev: "mp.Event",
):
    """
    Worker: opens its own memmap, samples batches on CPU, and pushes numpy arrays.
    We push numpy arrays (not torch tensors) to reduce pickling overhead sometimes.
    """
    rng = np.random.default_rng(seed)
    tokens = np.load(tokens_path, mmap_mode="r")
    n = int(tokens.shape[0])
    max_start = n - context_length - 1
    if max_start <= 0:
        raise ValueError("Token array too small for given context_length")

    while not stop_ev.is_set():
        starts = rng.integers(0, max_start, size=batch_size, endpoint=False)

        x_np = np.empty((batch_size, context_length), dtype=np.int64)
        y_np = np.empty((batch_size, context_length), dtype=np.int64)

        for i, s in enumerate(starts):
            x_np[i] = tokens[s : s + context_length]
            y_np[i] = tokens[s + 1 : s + 1 + context_length]

        out_q.put((x_np, y_np))


class BatchPrefetcher:
    def __init__(
        self,
        tokens_path: str,
        batch_size: int,
        context_length: int,
        num_workers: int = 2,
        prefetch: int = 8,
        base_seed: int = 1234,
    ):
        self.ctx = mp.get_context("spawn")
        self.stop_ev = self.ctx.Event()
        self.q: mp.Queue = self.ctx.Queue(maxsize=prefetch)
        self.workers: list[mp.Process] = []

        for i in range(num_workers):
            p = self.ctx.Process(
                target=_batch_worker,
                args=(tokens_path, batch_size, context_length, base_seed + i, self.q, self.stop_ev),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        return self.q.get()

    def close(self):
        self.stop_ev.set()
        for p in self.workers:
            p.join(timeout=2.0)
        while True:
            try:
                self.q.get_nowait()
            except Exception:
                break

@dataclass
class TrainConfig:

    train_tokens_path: str
    val_tokens_path: Optional[str]
    vocab_size: int
    context_length: int
    batch_size: int

    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float

    max_lr: float
    min_lr: float
    warmup_iters: int
    cosine_cycle_iters: int
    betas1: float
    betas2: float
    eps: float
    weight_decay: float
    grad_clip: float

    max_iters: int
    log_every: int
    eval_every: int
    eval_batches: int

    ckpt_path: Optional[str]
    save_every: int
    resume: bool

    prefetch_workers: int
    prefetch_depth: int
    heartbeat_every_s: float

    device: str
    dtype: str


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def pick_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def load_memmap_tokens(path: str) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D token array, got shape={arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Expected integer tokens, got dtype={arr.dtype}")
    return arr


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokens: np.ndarray,
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    losses = []

    n = int(tokens.shape[0])
    max_start = n - cfg.context_length - 1
    rng = np.random.default_rng(0)

    for _ in range(cfg.eval_batches):
        starts = rng.integers(0, max_start, size=cfg.batch_size, endpoint=False)

        x_np = np.empty((cfg.batch_size, cfg.context_length), dtype=np.int64)
        y_np = np.empty((cfg.batch_size, cfg.context_length), dtype=np.int64)
        for i, s in enumerate(starts):
            x_np[i] = tokens[s : s + cfg.context_length]
            y_np[i] = tokens[s + 1 : s + 1 + cfg.context_length]

        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)

        logits = model(x)
        loss = cross_entropy_loss(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )
        losses.append(float(loss.detach().cpu()))

    avg_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    return avg_loss, ppl


def train(cfg: TrainConfig) -> None:
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype)
    print(f"[device] {device} | [dtype] {dtype}")

    val_tokens = load_memmap_tokens(cfg.val_tokens_path) if cfg.val_tokens_path else None

    model = TransformerLM(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        rope_theta=cfg.rope_theta,
        device=device,
        dtype=dtype,
    )

    opt = AdamW(
        model.parameters(),
        lr=cfg.max_lr,
        betas=(cfg.betas1, cfg.betas2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    start_it = 0
    if cfg.resume and cfg.ckpt_path and Path(cfg.ckpt_path).exists():
        start_it = load_checkpoint(cfg.ckpt_path, model, opt)
        print(f"[resume] loaded checkpoint from {cfg.ckpt_path}, iteration={start_it}")

    model.train()

    prefetcher = BatchPrefetcher(
        tokens_path=cfg.train_tokens_path,
        batch_size=cfg.batch_size,
        context_length=cfg.context_length,
        num_workers=cfg.prefetch_workers,
        prefetch=cfg.prefetch_depth,
    )
    print(f"[prefetch] workers={cfg.prefetch_workers} depth={cfg.prefetch_depth}")

    t_log = time.time()
    t_hb = time.time()
    hb_loss_sum = 0.0
    hb_loss_n = 0

    try:
        for it in range(start_it, cfg.max_iters):
           
            lr = get_lr_cosine_schedule(
                it=it,
                max_learning_rate=cfg.max_lr,
                min_learning_rate=cfg.min_lr,
                warmup_iters=cfg.warmup_iters,
                cosine_cycle_iters=cfg.cosine_cycle_iters,
            )
            for group in opt.param_groups:
                group["lr"] = lr

           
            x_np, y_np = prefetcher.get()
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_np).to(device)

            
            logits = model(x)
            loss = cross_entropy_loss(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            
            opt.zero_grad(set_to_none=True)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                clip_grad_l2(model.parameters(), cfg.grad_clip)

            opt.step()

            
            hb_loss_sum += float(loss.detach().cpu())
            hb_loss_n += 1
            now = time.time()
            if now - t_hb >= cfg.heartbeat_every_s:
                pct = 100.0 * (it + 1) / cfg.max_iters
                recent_avg = hb_loss_sum / max(1, hb_loss_n)
                print(f"[hb] it={it+1} ({pct:.1f}%) | recent_avg_loss={recent_avg:.4f} | lr={lr:.6g}")
                t_hb = now
                hb_loss_sum = 0.0
                hb_loss_n = 0

           
            if (it + 1) % cfg.log_every == 0:
                dt = time.time() - t_log
                toks = cfg.batch_size * cfg.context_length * cfg.log_every
                toks_per_s = toks / max(1e-9, dt)
                l = float(loss.detach().cpu())
                print(
                    f"it={it+1:>7} | loss={l:.4f} | ppl={math.exp(l):.2f} "
                    f"| lr={lr:.6g} | tok/s={toks_per_s:.1f}"
                )
                t_log = time.time()

            
            if val_tokens is not None and (it + 1) % cfg.eval_every == 0:
                val_loss, val_ppl = evaluate(model, val_tokens, cfg, device)
                print(f"[val] it={it+1:>7} | loss={val_loss:.4f} | ppl={val_ppl:.2f}")
                model.train()

           
            if cfg.ckpt_path and ((it + 1) % cfg.save_every == 0 or (it + 1) == cfg.max_iters):
                Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(model, opt, it + 1, cfg.ckpt_path)
                print(f"[ckpt] saved to {cfg.ckpt_path} (it={it+1})")

    finally:
        prefetcher.close()


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser("Train Transformer LM")

    # data
    p.add_argument("--train_tokens_path", type=str, required=True)
    p.add_argument("--val_tokens_path", type=str, default=None)
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)

    # model
    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    # optim
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--cosine_cycle_iters", type=int, default=5000)
    p.add_argument("--betas1", type=float, default=0.9)
    p.add_argument("--betas2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # loop
    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=10)

    # checkpointing
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--resume", action="store_true")

    # perf
    p.add_argument("--prefetch_workers", type=int, default=2)
    p.add_argument("--prefetch_depth", type=int, default=8)
    p.add_argument("--heartbeat_every_s", type=float, default=10.0)

    # device / dtype
    p.add_argument("--device", type=str, default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--dtype", type=str, default="float32", help="float32 | float16 | bfloat16")

    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)