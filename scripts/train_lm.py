from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import sys
import time
from typing import Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import TransformerLM
from training.training import (
    AdamW,
    clip_grad_l2,
    cross_entropy_loss,
    get_lr_cosine_schedule,
    load_checkpoint,
    save_checkpoint,
)
from training.wandb_utils import (
    configure_wandb_environment,
    download_artifact,
    infer_vocab_size_from_vocab_json,
    init_wandb_run,
    log_checkpoint_artifact,
    parse_csv_values,
    resolve_dataset_files,
    wandb_is_enabled,
)


def _batch_worker(
    tokens_path: str,
    batch_size: int,
    context_length: int,
    seed: int,
    out_q: "mp.Queue[Tuple[np.ndarray, np.ndarray]]",
    stop_ev: "mp.Event",
):
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
    train_tokens_path: Optional[str]
    val_tokens_path: Optional[str]
    vocab_size: Optional[int]
    vocab_json_path: Optional[str]
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
    target_tokens_seen: Optional[int]
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

    scratch_dir: Optional[str]

    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_mode: str
    wandb_run_name: Optional[str]
    wandb_group: Optional[str]
    wandb_tags: Optional[str]
    wandb_dir: Optional[str]
    wandb_cache_dir: Optional[str]
    wandb_artifact_dir: Optional[str]
    wandb_data_dir: Optional[str]

    dataset_artifact: Optional[str]
    dataset_download_dir: Optional[str]
    dataset_train_name: str
    dataset_val_name: str
    dataset_vocab_name: str
    dataset_merges_name: str

    resume_artifact: Optional[str]
    resume_download_dir: Optional[str]

    checkpoint_artifact_name: Optional[str]
    checkpoint_keep_milestone_every: int
    checkpoint_ttl_days: Optional[int]

    run_record_dir: Optional[str]


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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _tokens_per_step(cfg: TrainConfig) -> int:
    return cfg.batch_size * cfg.context_length


def _resolve_training_budget(cfg: TrainConfig) -> None:
    if cfg.target_tokens_seen is None:
        return

    tokens_per_step = _tokens_per_step(cfg)
    resolved_max_iters = math.ceil(cfg.target_tokens_seen / tokens_per_step)
    cfg.max_iters = max(1, resolved_max_iters)
    print(
        f"[budget] target_tokens_seen={cfg.target_tokens_seen} "
        f"| tokens_per_step={tokens_per_step} | max_iters={cfg.max_iters}"
    )


def _write_json(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _default_checkpoint_path(cfg: TrainConfig) -> str:
    base_dir = Path(cfg.scratch_dir).expanduser().resolve() if cfg.scratch_dir else PROJECT_ROOT
    ckpt_dir = base_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return str((ckpt_dir / "train_lm.pt").resolve())


def _best_checkpoint_path(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    return str(path.with_name(f"{path.stem}.best{path.suffix}"))


def _default_dataset_download_dir(cfg: TrainConfig) -> str:
    if cfg.scratch_dir:
        return str((Path(cfg.scratch_dir).expanduser().resolve() / "datasets").resolve())
    return str((PROJECT_ROOT / "data" / "wandb_dataset").resolve())


def _default_resume_download_dir(cfg: TrainConfig) -> str:
    if cfg.scratch_dir:
        return str((Path(cfg.scratch_dir).expanduser().resolve() / "resume_artifacts").resolve())
    return str((PROJECT_ROOT / "checkpoints" / "resume_artifacts").resolve())


def _validate_cfg(cfg: TrainConfig) -> None:
    if not cfg.train_tokens_path and not cfg.dataset_artifact:
        raise ValueError("Provide either --train_tokens_path or --dataset_artifact")
    if wandb_is_enabled(cfg.wandb_mode) and not cfg.wandb_project:
        raise ValueError("--wandb_project is required when W&B mode is enabled")
    if cfg.resume and not (cfg.ckpt_path or cfg.resume_artifact):
        raise ValueError("--resume requires either --ckpt_path or --resume_artifact")


def _resolve_dataset_inputs(
    cfg: TrainConfig,
    wandb_run: object | None,
) -> tuple[str, Optional[str], Optional[str]]:
    train_tokens_path = cfg.train_tokens_path
    val_tokens_path = cfg.val_tokens_path
    vocab_json_path = cfg.vocab_json_path

    if cfg.dataset_artifact:
        dataset_download_dir = cfg.dataset_download_dir or _default_dataset_download_dir(cfg)
        print(f"[dataset] downloading artifact {cfg.dataset_artifact} -> {dataset_download_dir}")
        _, artifact_dir = download_artifact(
            artifact_ref=cfg.dataset_artifact,
            root=dataset_download_dir,
            type_name="dataset",
            run=wandb_run,
        )
        files = resolve_dataset_files(
            artifact_dir,
            train_name=cfg.dataset_train_name,
            val_name=cfg.dataset_val_name,
            vocab_name=cfg.dataset_vocab_name,
            merges_name=cfg.dataset_merges_name,
        )
        train_tokens_path = str(files["train_tokens_path"])
        val_tokens_path = str(files["val_tokens_path"])
        vocab_json_path = str(files["vocab_path"])
        print(f"[dataset] using staged files from {artifact_dir}")

    if not train_tokens_path:
        raise ValueError("No train tokens path resolved")

    if cfg.vocab_size is None:
        if not vocab_json_path:
            raise ValueError("Provide --vocab_size or a vocab.json via --vocab_json_path / --dataset_artifact")
        cfg.vocab_size = infer_vocab_size_from_vocab_json(vocab_json_path)
        print(f"[vocab] inferred vocab_size={cfg.vocab_size} from {vocab_json_path}")
    elif vocab_json_path:
        inferred_vocab_size = infer_vocab_size_from_vocab_json(vocab_json_path)
        if inferred_vocab_size != cfg.vocab_size:
            raise ValueError(
                f"Provided vocab_size={cfg.vocab_size} does not match "
                f"vocab.json size={inferred_vocab_size}"
            )

    return train_tokens_path, val_tokens_path, vocab_json_path


def _resolve_resume_path(
    cfg: TrainConfig,
    wandb_run: object | None,
) -> Optional[str]:
    resume_checkpoint_path = cfg.ckpt_path

    if cfg.resume_artifact:
        resume_download_dir = cfg.resume_download_dir or _default_resume_download_dir(cfg)
        print(f"[resume] downloading checkpoint artifact {cfg.resume_artifact} -> {resume_download_dir}")
        _, artifact_dir = download_artifact(
            artifact_ref=cfg.resume_artifact,
            root=resume_download_dir,
            type_name="model",
            run=wandb_run,
        )
        artifact_checkpoint = artifact_dir / "checkpoint.pt"
        if not artifact_checkpoint.exists():
            raise FileNotFoundError(f"checkpoint.pt not found in artifact download {artifact_dir}")
        resume_checkpoint_path = str(artifact_checkpoint)
        cfg.resume = True
        print(f"[resume] using checkpoint artifact file {resume_checkpoint_path}")

    return resume_checkpoint_path


def _maybe_log_checkpoint_artifact(
    *,
    wandb_run: object | None,
    artifact_name: str | None,
    checkpoint_path: str | None,
    aliases: list[str],
    metadata: dict[str, object],
    ttl_days: int | None,
) -> None:
    if wandb_run is None or artifact_name is None or checkpoint_path is None:
        return
    if not aliases:
        return
    log_checkpoint_artifact(
        run=wandb_run,
        artifact_name=artifact_name,
        checkpoint_path=checkpoint_path,
        aliases=aliases,
        metadata=metadata,
        ttl_days=ttl_days,
    )
    print(f"[wandb] logged checkpoint artifact {artifact_name} aliases={','.join(aliases)}")


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
    _resolve_training_budget(cfg)
    _validate_cfg(cfg)

    run_record_dir = Path(cfg.run_record_dir).expanduser().resolve() if cfg.run_record_dir else None
    if run_record_dir is not None:
        run_record_dir.mkdir(parents=True, exist_ok=True)

    configure_wandb_environment(
        scratch_dir=cfg.scratch_dir,
        wandb_dir=cfg.wandb_dir,
        wandb_cache_dir=cfg.wandb_cache_dir,
        wandb_artifact_dir=cfg.wandb_artifact_dir,
        wandb_data_dir=cfg.wandb_data_dir,
    )

    wandb_run = None
    if wandb_is_enabled(cfg.wandb_mode):
        wandb_run = init_wandb_run(
            project=cfg.wandb_project or "",
            entity=cfg.wandb_entity,
            mode=cfg.wandb_mode,
            job_type="train",
            name=cfg.wandb_run_name,
            group=cfg.wandb_group,
            tags=parse_csv_values(cfg.wandb_tags),
            config=asdict(cfg),
        )

    try:
        train_tokens_path, val_tokens_path, vocab_json_path = _resolve_dataset_inputs(cfg, wandb_run)
        resume_checkpoint_path = _resolve_resume_path(cfg, wandb_run)

        if cfg.ckpt_path is None and (cfg.resume or wandb_run is not None or cfg.resume_artifact):
            cfg.ckpt_path = _default_checkpoint_path(cfg)
            print(f"[ckpt] default local checkpoint path {cfg.ckpt_path}")

        device = pick_device(cfg.device)
        dtype = pick_dtype(cfg.dtype)
        print(f"[device] {device} | [dtype] {dtype}")

        val_tokens = load_memmap_tokens(val_tokens_path) if val_tokens_path else None

        assert cfg.vocab_size is not None
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
        parameter_count = count_parameters(model)
        tokens_per_step = _tokens_per_step(cfg)
        print(f"[model] parameters={parameter_count} | tokens_per_step={tokens_per_step}")

        if wandb_run is not None:
            wandb_run.config.update(
                {
                    "resolved_train_tokens_path": train_tokens_path,
                    "resolved_val_tokens_path": val_tokens_path,
                    "resolved_vocab_json_path": vocab_json_path,
                    "resolved_ckpt_path": cfg.ckpt_path,
                    "resolved_vocab_size": cfg.vocab_size,
                    "resolved_max_iters": cfg.max_iters,
                    "resolved_target_tokens_seen": cfg.target_tokens_seen,
                    "model/parameter_count": parameter_count,
                    "train/tokens_per_step": tokens_per_step,
                },
                allow_val_change=True,
            )
            wandb_run.summary["model/parameter_count"] = parameter_count
            wandb_run.summary["train/tokens_per_step"] = tokens_per_step

        if run_record_dir is not None:
            _write_json(
                run_record_dir / "resolved_config.json",
                {
                    **asdict(cfg),
                    "resolved_train_tokens_path": train_tokens_path,
                    "resolved_val_tokens_path": val_tokens_path,
                    "resolved_vocab_json_path": vocab_json_path,
                    "resolved_resume_checkpoint_path": resume_checkpoint_path,
                    "resolved_ckpt_path": cfg.ckpt_path,
                    "parameter_count": parameter_count,
                    "tokens_per_step": tokens_per_step,
                },
            )

        opt = AdamW(
            model.parameters(),
            lr=cfg.max_lr,
            betas=(cfg.betas1, cfg.betas2),
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )

        start_it = 0
        if cfg.resume and resume_checkpoint_path and Path(resume_checkpoint_path).exists():
            start_it = load_checkpoint(resume_checkpoint_path, model, opt)
            print(f"[resume] loaded checkpoint from {resume_checkpoint_path}, iteration={start_it}")
        if start_it >= cfg.max_iters:
            raise ValueError(
                f"Resume checkpoint iteration {start_it} is >= max_iters {cfg.max_iters}. "
                "Set --max_iters to the total final training step you want to reach."
            )

        model.train()

        prefetcher = BatchPrefetcher(
            tokens_path=train_tokens_path,
            batch_size=cfg.batch_size,
            context_length=cfg.context_length,
            num_workers=cfg.prefetch_workers,
            prefetch=cfg.prefetch_depth,
        )
        print(f"[prefetch] workers={cfg.prefetch_workers} depth={cfg.prefetch_depth}")

        checkpoint_artifact_name = cfg.checkpoint_artifact_name
        if checkpoint_artifact_name is None and wandb_run is not None:
            checkpoint_artifact_name = f"transformer-checkpoints-{wandb_run.id}"

        t_log = time.time()
        t_hb = time.time()
        hb_loss_sum = 0.0
        hb_loss_n = 0
        best_val_loss: float | None = None
        best_val_ppl: float | None = None
        best_iteration: int | None = None
        train_start_time = time.time()
        last_step = start_it
        last_train_loss: float | None = None
        last_train_ppl: float | None = None
        last_tokens_per_second: float | None = None

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

                step = it + 1
                last_step = step
                hb_loss_sum += float(loss.detach().cpu())
                hb_loss_n += 1
                now = time.time()
                if now - t_hb >= cfg.heartbeat_every_s:
                    pct = 100.0 * step / cfg.max_iters
                    recent_avg = hb_loss_sum / max(1, hb_loss_n)
                    tokens_seen = step * tokens_per_step
                    wall_clock_seconds = now - train_start_time
                    flops_proxy = float(6 * parameter_count * tokens_seen)
                    print(f"[hb] it={step} ({pct:.1f}%) | recent_avg_loss={recent_avg:.4f} | lr={lr:.6g}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "heartbeat/recent_avg_loss": recent_avg,
                                "train/lr": lr,
                                "train/tokens_seen": tokens_seen,
                                "perf/wall_clock_seconds": wall_clock_seconds,
                                "perf/flops_proxy": flops_proxy,
                                "model/parameter_count": parameter_count,
                            },
                            step=step,
                        )
                    t_hb = now
                    hb_loss_sum = 0.0
                    hb_loss_n = 0

                if step % cfg.log_every == 0:
                    dt = time.time() - t_log
                    toks = cfg.batch_size * cfg.context_length * cfg.log_every
                    toks_per_s = toks / max(1e-9, dt)
                    loss_value = float(loss.detach().cpu())
                    last_train_loss = loss_value
                    last_train_ppl = math.exp(loss_value)
                    last_tokens_per_second = toks_per_s
                    tokens_seen = step * tokens_per_step
                    wall_clock_seconds = time.time() - train_start_time
                    flops_proxy = float(6 * parameter_count * tokens_seen)
                    print(
                        f"it={step:>7} | loss={loss_value:.4f} | ppl={math.exp(loss_value):.2f} "
                        f"| lr={lr:.6g} | tok/s={toks_per_s:.1f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train/loss": loss_value,
                                "train/ppl": math.exp(loss_value),
                                "train/lr": lr,
                                "train/tokens_seen": tokens_seen,
                                "perf/tokens_per_second": toks_per_s,
                                "perf/wall_clock_seconds": wall_clock_seconds,
                                "perf/flops_proxy": flops_proxy,
                                "model/parameter_count": parameter_count,
                            },
                            step=step,
                        )
                    t_log = time.time()

                if val_tokens is not None and step % cfg.eval_every == 0:
                    val_loss, val_ppl = evaluate(model, val_tokens, cfg, device)
                    tokens_seen = step * tokens_per_step
                    wall_clock_seconds = time.time() - train_start_time
                    flops_proxy = float(6 * parameter_count * tokens_seen)
                    print(f"[val] it={step:>7} | loss={val_loss:.4f} | ppl={val_ppl:.2f}")
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "val/loss": val_loss,
                                "val/ppl": val_ppl,
                                "train/tokens_seen": tokens_seen,
                                "perf/wall_clock_seconds": wall_clock_seconds,
                                "perf/flops_proxy": flops_proxy,
                                "model/parameter_count": parameter_count,
                            },
                            step=step,
                        )

                    improved = best_val_loss is None or val_loss < best_val_loss
                    if improved:
                        best_val_loss = val_loss
                        best_val_ppl = val_ppl
                        best_iteration = step
                        print(f"[best] it={step} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")
                        if wandb_run is not None:
                            wandb_run.log(
                                {
                                    "best/val_loss": best_val_loss,
                                    "best/val_ppl": best_val_ppl,
                                    "best/iteration": best_iteration,
                                },
                                step=step,
                            )
                        if cfg.ckpt_path:
                            best_ckpt_path = _best_checkpoint_path(cfg.ckpt_path)
                            Path(best_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                            save_checkpoint(model, opt, step, best_ckpt_path)
                            print(f"[best] saved checkpoint to {best_ckpt_path}")
                            _maybe_log_checkpoint_artifact(
                                wandb_run=wandb_run,
                                artifact_name=checkpoint_artifact_name,
                                checkpoint_path=best_ckpt_path,
                                aliases=["best", f"best-step-{step:07d}"],
                                metadata={
                                    "iteration": step,
                                    "kind": "best",
                                    "val_loss": val_loss,
                                    "val_ppl": val_ppl,
                                    "train_tokens_path": train_tokens_path,
                                },
                                ttl_days=cfg.checkpoint_ttl_days,
                            )
                    model.train()

                should_save = cfg.ckpt_path and (step % cfg.save_every == 0 or step == cfg.max_iters)
                if should_save:
                    Path(cfg.ckpt_path).parent.mkdir(parents=True, exist_ok=True)
                    save_checkpoint(model, opt, step, cfg.ckpt_path)
                    print(f"[ckpt] saved to {cfg.ckpt_path} (it={step})")

                    aliases = ["latest"]
                    if cfg.checkpoint_keep_milestone_every > 0 and step % cfg.checkpoint_keep_milestone_every == 0:
                        aliases.append(f"step-{step:07d}")
                    if step == cfg.max_iters:
                        aliases.append("final")

                    _maybe_log_checkpoint_artifact(
                        wandb_run=wandb_run,
                        artifact_name=checkpoint_artifact_name,
                        checkpoint_path=cfg.ckpt_path,
                        aliases=aliases,
                        metadata={
                            "iteration": step,
                            "kind": "latest",
                            "train_tokens_path": train_tokens_path,
                            "val_tokens_path": val_tokens_path,
                            "vocab_size": cfg.vocab_size,
                            "best_val_loss": best_val_loss,
                            "best_val_ppl": best_val_ppl,
                            "best_iteration": best_iteration,
                        },
                        ttl_days=cfg.checkpoint_ttl_days,
                    )

        finally:
            prefetcher.close()
            if run_record_dir is not None:
                total_tokens_seen = last_step * tokens_per_step
                _write_json(
                    run_record_dir / "summary.json",
                    {
                        "completed_iterations": last_step,
                        "max_iters": cfg.max_iters,
                        "target_tokens_seen": cfg.target_tokens_seen,
                        "tokens_per_step": tokens_per_step,
                        "tokens_seen": total_tokens_seen,
                        "parameter_count": parameter_count,
                        "flops_proxy": float(6 * parameter_count * total_tokens_seen),
                        "wall_clock_seconds": time.time() - train_start_time,
                        "last_train_loss": last_train_loss,
                        "last_train_ppl": last_train_ppl,
                        "last_tokens_per_second": last_tokens_per_second,
                        "best_val_loss": best_val_loss,
                        "best_val_ppl": best_val_ppl,
                        "best_iteration": best_iteration,
                        "wandb_run_id": getattr(wandb_run, "id", None),
                        "wandb_run_name": getattr(wandb_run, "name", None),
                        "wandb_run_url": getattr(wandb_run, "url", None),
                    },
                )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser("Train Transformer LM")

    p.add_argument("--train_tokens_path", type=str, default=None)
    p.add_argument("--val_tokens_path", type=str, default=None)
    p.add_argument("--vocab_size", type=int, default=None)
    p.add_argument("--vocab_json_path", type=str, default=None)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)

    p.add_argument("--d_model", type=int, default=384)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--rope_theta", type=float, default=10000.0)

    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--cosine_cycle_iters", type=int, default=5000)
    p.add_argument("--betas1", type=float, default=0.9)
    p.add_argument("--betas2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--max_iters", type=int, default=5000)
    p.add_argument("--target_tokens_seen", type=int, default=None, help="Optional total training tokens to process; overrides max_iters")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_batches", type=int, default=10)

    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--resume", action="store_true")

    p.add_argument("--prefetch_workers", type=int, default=2)
    p.add_argument("--prefetch_depth", type=int, default=8)
    p.add_argument("--heartbeat_every_s", type=float, default=10.0)

    p.add_argument("--device", type=str, default="auto", help="auto | cpu | mps | cuda")
    p.add_argument("--dtype", type=str, default="float32", help="float32 | float16 | bfloat16")

    p.add_argument("--scratch_dir", type=str, default=None, help="Optional scratch root for checkpoints and W&B files")

    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_mode", type=str, default="disabled", help="online | offline | disabled")
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None, help="Comma-separated W&B tags")
    p.add_argument("--wandb_dir", type=str, default=None)
    p.add_argument("--wandb_cache_dir", type=str, default=None)
    p.add_argument("--wandb_artifact_dir", type=str, default=None)
    p.add_argument("--wandb_data_dir", type=str, default=None)

    p.add_argument("--dataset_artifact", type=str, default=None, help="Dataset artifact ref, e.g. entity/project/name:latest")
    p.add_argument("--dataset_download_dir", type=str, default=None)
    p.add_argument("--dataset_train_name", type=str, default="train_tokens_full_w8.npy")
    p.add_argument("--dataset_val_name", type=str, default="val_tokens_full_w8.npy")
    p.add_argument("--dataset_vocab_name", type=str, default="vocab.json")
    p.add_argument("--dataset_merges_name", type=str, default="merges.json")

    p.add_argument("--resume_artifact", type=str, default=None, help="Checkpoint artifact ref, e.g. entity/project/name:latest")
    p.add_argument("--resume_download_dir", type=str, default=None)

    p.add_argument("--checkpoint_artifact_name", type=str, default=None, help="Artifact collection name for uploaded checkpoints")
    p.add_argument("--checkpoint_keep_milestone_every", type=int, default=0, help="Add step-XXXX aliases every N steps")
    p.add_argument("--checkpoint_ttl_days", type=int, default=None, help="Optional TTL in days for W&B-hosted checkpoints")
    p.add_argument("--run_record_dir", type=str, default=None, help="Optional local directory where resolved config and summary JSON will be written")

    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
