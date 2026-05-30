"""Capability-suite orchestrator.

Loads a trained checkpoint + its resolved_config.json, reconstructs the model
via `model.build_model`, runs every capability probe in `eval/`, and writes a
single JSON of metrics. Optionally logs to W&B.

Designed to be called once per training run by the architecture-comparison
sweep, e.g.

    python3 eval/run_capability_suite.py \\
        --checkpoint /tmp/.../checkpoints/train_lm.best.pt \\
        --resolved_config sweep_experiments/.../arch_run/resolved_config.json \\
        --val_tokens_path data/val_tokens_full_w8.npy \\
        --output_json sweep_experiments/.../arch_run/capability.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import build_model
from eval import baselines, induction, long_context, needle_haystack, parens, selective_copy, story_quality


def _pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _pick_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _kwargs_from_resolved(cfg: dict) -> dict[str, Any]:
    """Pull the build_model kwargs out of a `resolved_config.json` payload."""
    keys = (
        "model_arch", "vocab_size", "context_length",
        "d_model", "num_layers", "num_heads", "d_ff", "rope_theta",
        "ssm_d_state", "ssm_d_conv", "ssm_expand",
        "subq_kind", "subq_window_size", "subq_feature_map",
    )
    out: dict[str, Any] = {}
    for k in keys:
        if k in cfg:
            out[k] = cfg[k]
    # Resolved vocab size sometimes lives under a different key.
    if "vocab_size" not in out and "resolved_vocab_size" in cfg:
        out["vocab_size"] = cfg["resolved_vocab_size"]
    return out


def _build_from_cfg(cfg_kwargs: dict[str, Any], device, dtype) -> torch.nn.Module:
    arch = cfg_kwargs.pop("model_arch", "transformer")
    return build_model(arch=arch, device=device, dtype=dtype, **cfg_kwargs)


def main() -> None:
    p = argparse.ArgumentParser("capability suite")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--resolved_config", type=str, required=True,
                   help="resolved_config.json written by train_lm.py")
    p.add_argument("--val_tokens_path", type=str, required=True)
    p.add_argument("--train_tokens_path", type=str, default=None,
                   help="Only needed for the bigram baseline.")
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32")
    p.add_argument("--vocab_size_from_resolved", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="If set, attach capability metrics to an existing W&B run.")
    args = p.parse_args()

    device = _pick_device(args.device)
    dtype = _pick_dtype(args.dtype)
    print(f"[capability] device={device} dtype={dtype}")

    cfg = json.loads(Path(args.resolved_config).read_text(encoding="utf-8"))
    build_kwargs = _kwargs_from_resolved(cfg)
    vocab_size = int(build_kwargs.get("vocab_size") or cfg.get("resolved_vocab_size") or 0)
    if vocab_size <= 0:
        raise ValueError("Could not determine vocab_size from resolved_config.json")
    build_kwargs["vocab_size"] = vocab_size

    # Build & load checkpoint.
    model = _build_from_cfg(dict(build_kwargs), device, dtype)
    payload = torch.load(args.checkpoint, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[capability] loaded checkpoint {args.checkpoint} ({n_params} params)")

    val_tokens = np.load(args.val_tokens_path, mmap_mode="r")
    if val_tokens.ndim != 1:
        raise ValueError(f"expected 1-D val tokens, got shape={val_tokens.shape}")

    context_length_train = int(cfg.get("context_length", getattr(model, "context_length", 256)))
    metrics: dict[str, float] = {}

    # --- Probes ---------------------------------------------------------
    print("[capability] long_context")
    metrics.update(
        long_context.run(
            model,
            val_tokens=val_tokens,
            context_length_train=context_length_train,
            eval_lengths=(context_length_train, context_length_train * 2, context_length_train * 4),
            device=device,
        )
    )
    print("[capability] selective_copy")
    metrics.update(selective_copy.run(model, vocab_size=vocab_size, device=device))
    print("[capability] induction")
    metrics.update(induction.run(model, vocab_size=vocab_size, device=device))
    print("[capability] needle_haystack")
    metrics.update(
        needle_haystack.run(model, val_tokens=val_tokens, vocab_size=vocab_size, device=device)
    )
    print("[capability] parens")
    metrics.update(parens.run(model, vocab_size=vocab_size, device=device))
    print("[capability] story_quality")
    metrics.update(story_quality.run(model, val_tokens=val_tokens, device=device))

    # --- Baselines ------------------------------------------------------
    metrics["baseline/random_acc"] = baselines.random_baseline_acc(vocab_size)
    metrics["baseline/random_ppl"] = baselines.random_baseline_ppl(vocab_size)
    if args.train_tokens_path and Path(args.train_tokens_path).exists():
        try:
            train_tokens = np.load(args.train_tokens_path, mmap_mode="r")
            metrics["baseline/bigram_ppl"] = baselines.bigram_baseline_ppl(
                np.asarray(train_tokens[: min(len(train_tokens), 2_000_000)]),
                np.asarray(val_tokens[: min(len(val_tokens), 200_000)]),
            )
        except Exception as exc:
            print(f"[capability] bigram baseline failed: {exc}")

    metrics["meta/arch"] = cfg.get("model_arch", "unknown")
    metrics["meta/parameter_count"] = float(n_params)
    metrics["meta/context_length_train"] = float(context_length_train)

    # --- Write & log ----------------------------------------------------
    # Sanitize non-finite floats (NaN/inf) to None so the output is valid JSON
    # consumable by strict parsers and W&B. A probe returns NaN when its config
    # exceeds the model's supported context (e.g. needle@middle128 on a 64-ctx
    # model) — those cells are intentionally skipped.
    def _clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    metrics_clean = {k: _clean(v) for k, v in metrics.items()}

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics_clean, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[capability] wrote {out_path}")

    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                id=args.wandb_run_id,
                resume="allow",
                reinit=True,
            )
            wandb.log(
                {
                    k: v
                    for k, v in metrics.items()
                    if isinstance(v, (int, float)) and math.isfinite(v)
                }
            )
            run.finish()
        except Exception as exc:
            print(f"[capability] W&B logging skipped: {exc}")


if __name__ == "__main__":
    main()
