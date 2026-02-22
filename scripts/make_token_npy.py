from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring import collect_tokenization_stats
from tokenizer.bpe import BPETrainer
from tokenizer.tokenizer import Tokenizer


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _normalize_worker_counts(worker_counts: list[int]) -> list[int]:
    normalized: list[int] = []
    seen: set[int] = set()
    for worker_count in worker_counts:
        if worker_count < 1:
            raise ValueError("All num_workers values must be >= 1")
        if worker_count not in seen:
            normalized.append(worker_count)
            seen.add(worker_count)
    return normalized


def _save_experiment_artifacts(
    experiment_dir: Path,
    tokenizer: Tokenizer,
    train_arr: np.ndarray,
    val_arr: np.ndarray,
    experiment_record: dict[str, Any],
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    np.save(experiment_dir / "train_tokens.npy", train_arr)
    np.save(experiment_dir / "val_tokens.npy", val_arr)
    _write_json(experiment_dir / "vocab.json", tokenizer.to_serializable_vocab())
    _write_json(experiment_dir / "merges.json", tokenizer.to_serializable_merges())
    _write_json(experiment_dir / "experiment.json", experiment_record)


def _describe_array(arr: np.ndarray) -> str:
    if arr.size == 0:
        return f"shape={arr.shape} dtype={arr.dtype} empty"
    return (
        f"shape={arr.shape} dtype={arr.dtype} "
        f"min={arr.min()} max={arr.max()}"
    )


def _run_experiment(
    worker_count: int,
    args: argparse.Namespace,
    special_tokens: list[str],
    train_text: str,
    val_text: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    trainer = BPETrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_workers=worker_count,
        pretoken_batch_size=args.training_batch_size,
    )

    training_start = time.perf_counter()
    vocab, merges = trainer.train(args.train_txt)
    training_elapsed = time.perf_counter() - training_start

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    train_arr, train_stats = collect_tokenization_stats(
        tokenizer=tokenizer,
        text=train_text,
        num_workers=worker_count,
        batch_size=args.tokenization_batch_size,
    )
    val_arr, val_stats = collect_tokenization_stats(
        tokenizer=tokenizer,
        text=val_text,
        num_workers=worker_count,
        batch_size=args.tokenization_batch_size,
    )

    experiment_dir = Path(args.experiments_dir) / f"numWorkers_{worker_count}"
    experiment_record: dict[str, Any] = {
        "num_workers": worker_count,
        "config": {
            "train_txt": args.train_txt,
            "val_txt": args.val_txt,
            "vocab_size": args.vocab_size,
            "special_tokens": special_tokens,
            "training_batch_size": args.training_batch_size,
            "tokenization_batch_size": args.tokenization_batch_size,
        },
        "training": {
            "duration_seconds": training_elapsed,
            "vocab_size": len(tokenizer.vocab),
            "merge_count": len(tokenizer.merges),
        },
        "splits": {
            "train": train_stats,
            "val": val_stats,
        },
        "artifacts": {
            "train_tokens_npy": str(experiment_dir / "train_tokens.npy"),
            "val_tokens_npy": str(experiment_dir / "val_tokens.npy"),
            "vocab_json": str(experiment_dir / "vocab.json"),
            "merges_json": str(experiment_dir / "merges.json"),
        },
    }

    _save_experiment_artifacts(
        experiment_dir=experiment_dir,
        tokenizer=tokenizer,
        train_arr=train_arr,
        val_arr=val_arr,
        experiment_record=experiment_record,
    )

    print(
        f"[experiment] workers={worker_count} "
        f"| train_bpe_s={training_elapsed:.3f} "
        f"| train_tokens={train_stats['num_tokens']} "
        f"| val_tokens={val_stats['num_tokens']} "
        f"| dir={experiment_dir}"
    )

    return train_arr, val_arr, experiment_record


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_txt", type=str, required=True)
    p.add_argument("--val_txt", type=str, required=True)
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument(
        "--num_workers",
        type=int,
        nargs="+",
        default=[1],
        help="One or more worker counts to run, e.g. --num_workers 1 2 4",
    )
    p.add_argument(
        "--special_tokens",
        type=str,
        default="<|endoftext|>",
        help="Comma-separated list, default includes <|endoftext|>",
    )
    p.add_argument(
        "--training_batch_size",
        type=int,
        default=50_000,
        help="Number of pretokens per BPE counting task",
    )
    p.add_argument(
        "--tokenization_batch_size",
        type=int,
        default=4_096,
        help="Number of pretokens per tokenization task",
    )
    p.add_argument(
        "--experiments_dir",
        type=str,
        default="experiments",
        help="Directory used to store experiment artifacts and metrics",
    )
    p.add_argument("--out_train_npy", type=str, default="data/train_tokens.npy")
    p.add_argument("--out_val_npy", type=str, default="data/val_tokens.npy")
    args = p.parse_args()

    special_tokens = [s.strip() for s in args.special_tokens.split(",") if s.strip()]
    worker_counts = _normalize_worker_counts(args.num_workers)

    train_text = Path(args.train_txt).read_text(encoding="utf-8", errors="ignore")
    val_text = Path(args.val_txt).read_text(encoding="utf-8", errors="ignore")

    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    experiment_records: list[dict[str, Any]] = []
    final_train_arr: np.ndarray | None = None
    final_val_arr: np.ndarray | None = None

    for worker_count in worker_counts:
        final_train_arr, final_val_arr, experiment_record = _run_experiment(
            worker_count=worker_count,
            args=args,
            special_tokens=special_tokens,
            train_text=train_text,
            val_text=val_text,
        )
        experiment_records.append(experiment_record)

    _write_json(
        experiments_dir / "summary.json",
        {"experiments": experiment_records},
    )

    if len(worker_counts) == 1 and final_train_arr is not None and final_val_arr is not None:
        Path(args.out_train_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.out_train_npy, final_train_arr)
        np.save(args.out_val_npy, final_val_arr)

        print(f"Saved train: {args.out_train_npy}  {_describe_array(final_train_arr)}")
        print(f"Saved val:   {args.out_val_npy}  {_describe_array(final_val_arr)}")
    else:
        print(
            f"Saved experiment artifacts under {experiments_dir} "
            f"for worker counts: {', '.join(str(count) for count in worker_counts)}"
        )


if __name__ == "__main__":
    main()
