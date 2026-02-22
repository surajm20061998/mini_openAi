from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from monitoring import tokenize_file_to_npy_and_collect_stats
from tokenizer.bpe import BPETrainer
from tokenizer.tokenizer import (
    DEFAULT_PROGRESS_INTERVAL_BYTES,
    DEFAULT_READ_CHUNK_SIZE_BYTES,
    Tokenizer,
)


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


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _make_progress_logger(worker_count: int):
    def log(event: dict[str, Any]) -> None:
        stage = str(event.get("stage", "progress"))

        if "bytes_processed" in event and "total_bytes" in event:
            bytes_processed = int(event["bytes_processed"])
            total_bytes = int(event["total_bytes"])
            pct = 100.0 if total_bytes == 0 else (100.0 * bytes_processed / total_bytes)
            print(
                f"[progress] workers={worker_count} | {stage} | "
                f"{pct:5.1f}% | {_format_bytes(bytes_processed)}/{_format_bytes(total_bytes)}"
            )
            return

        if "completed_merges" in event and "total_merges" in event:
            completed_merges = int(event["completed_merges"])
            total_merges = int(event["total_merges"])
            pct = 100.0 if total_merges == 0 else (100.0 * completed_merges / total_merges)
            print(
                f"[progress] workers={worker_count} | {stage} | "
                f"{completed_merges}/{total_merges} merges ({pct:5.1f}%)"
            )
            return

        if stage == "bpe_count_complete":
            print(
                f"[progress] workers={worker_count} | {stage} | "
                f"unique_words={event.get('unique_words', 0)}"
            )

    return log


def _save_experiment_artifacts(
    experiment_dir: Path,
    tokenizer: Tokenizer,
    experiment_record: dict[str, Any],
) -> None:
    experiment_dir.mkdir(parents=True, exist_ok=True)
    _write_json(experiment_dir / "vocab.json", tokenizer.to_serializable_vocab())
    _write_json(experiment_dir / "merges.json", tokenizer.to_serializable_merges())
    _write_json(experiment_dir / "experiment.json", experiment_record)


def _describe_tokens(stats: dict[str, Any]) -> str:
    num_tokens = int(stats["num_tokens"])
    if num_tokens == 0:
        return "shape=(0,) dtype=int32 empty"

    return (
        f"shape=({num_tokens},) dtype=int32 "
        f"min={stats['token_id_min']} max={stats['token_id_max']}"
    )


def _run_experiment(
    worker_count: int,
    args: argparse.Namespace,
    special_tokens: list[str],
) -> tuple[Path, Path, dict[str, Any]]:
    experiment_dir = Path(args.experiments_dir) / f"numWorkers_{worker_count}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    progress_logger = _make_progress_logger(worker_count)

    trainer = BPETrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_workers=worker_count,
        pretoken_batch_size=args.training_batch_size,
    )

    print(
        f"[stage] workers={worker_count} | bpe_train | input={args.train_txt} "
        f"| vocab_size={args.vocab_size}"
    )
    training_start = time.perf_counter()
    vocab, merges = trainer.train(
        input_path=args.train_txt,
        read_chunk_size_bytes=args.read_chunk_size_bytes,
        progress_interval_bytes=args.progress_interval_bytes,
        progress_callback=progress_logger,
        merge_progress_every=args.merge_progress_every,
    )
    training_elapsed = time.perf_counter() - training_start
    print(
        f"[done] workers={worker_count} | bpe_train | {training_elapsed:.2f}s "
        f"| merges={len(merges)} | vocab={len(vocab)}"
    )

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    train_output_path = experiment_dir / "train_tokens.npy"
    print(
        f"[stage] workers={worker_count} | train_tokenize | input={args.train_txt} "
        f"| output={train_output_path}"
    )
    train_stats = tokenize_file_to_npy_and_collect_stats(
        tokenizer=tokenizer,
        input_path=args.train_txt,
        output_path=train_output_path,
        num_workers=worker_count,
        batch_size=args.tokenization_batch_size,
        read_chunk_size_bytes=args.read_chunk_size_bytes,
        progress_interval_bytes=args.progress_interval_bytes,
        progress_callback=progress_logger,
        progress_stage="train_tokenize",
        token_flush_size=args.token_flush_size,
        copy_chunk_tokens=args.copy_chunk_tokens,
    )
    print(
        f"[done] workers={worker_count} | train_tokenize | "
        f"{train_stats['tokenization_speed']['elapsed_seconds']:.2f}s "
        f"| tokens={train_stats['num_tokens']}"
    )

    val_output_path = experiment_dir / "val_tokens.npy"
    print(
        f"[stage] workers={worker_count} | val_tokenize | input={args.val_txt} "
        f"| output={val_output_path}"
    )
    val_stats = tokenize_file_to_npy_and_collect_stats(
        tokenizer=tokenizer,
        input_path=args.val_txt,
        output_path=val_output_path,
        num_workers=worker_count,
        batch_size=args.tokenization_batch_size,
        read_chunk_size_bytes=args.read_chunk_size_bytes,
        progress_interval_bytes=args.progress_interval_bytes,
        progress_callback=progress_logger,
        progress_stage="val_tokenize",
        token_flush_size=args.token_flush_size,
        copy_chunk_tokens=args.copy_chunk_tokens,
    )
    print(
        f"[done] workers={worker_count} | val_tokenize | "
        f"{val_stats['tokenization_speed']['elapsed_seconds']:.2f}s "
        f"| tokens={val_stats['num_tokens']}"
    )

    experiment_record: dict[str, Any] = {
        "num_workers": worker_count,
        "config": {
            "train_txt": args.train_txt,
            "val_txt": args.val_txt,
            "vocab_size": args.vocab_size,
            "special_tokens": special_tokens,
            "training_batch_size": args.training_batch_size,
            "tokenization_batch_size": args.tokenization_batch_size,
            "read_chunk_size_bytes": args.read_chunk_size_bytes,
            "progress_interval_bytes": args.progress_interval_bytes,
            "token_flush_size": args.token_flush_size,
            "copy_chunk_tokens": args.copy_chunk_tokens,
            "merge_progress_every": args.merge_progress_every,
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
            "train_tokens_npy": str(train_output_path),
            "val_tokens_npy": str(val_output_path),
            "vocab_json": str(experiment_dir / "vocab.json"),
            "merges_json": str(experiment_dir / "merges.json"),
        },
    }

    _save_experiment_artifacts(
        experiment_dir=experiment_dir,
        tokenizer=tokenizer,
        experiment_record=experiment_record,
    )

    print(
        f"[experiment] workers={worker_count} "
        f"| train_bpe_s={training_elapsed:.3f} "
        f"| train_tokens={train_stats['num_tokens']} "
        f"| val_tokens={val_stats['num_tokens']} "
        f"| dir={experiment_dir}"
    )

    return train_output_path, val_output_path, experiment_record


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
        "--read_chunk_size_bytes",
        type=int,
        default=DEFAULT_READ_CHUNK_SIZE_BYTES,
        help="File read chunk size used for streaming BPE training and tokenization",
    )
    p.add_argument(
        "--progress_interval_bytes",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL_BYTES,
        help="How often to print streaming progress; set <= 0 to disable",
    )
    p.add_argument(
        "--token_flush_size",
        type=int,
        default=4_194_304,
        help="How many token IDs to buffer before flushing to disk",
    )
    p.add_argument(
        "--copy_chunk_tokens",
        type=int,
        default=4_194_304,
        help="How many token IDs to copy per chunk when finalizing .npy files",
    )
    p.add_argument(
        "--merge_progress_every",
        type=int,
        default=100,
        help="How often to print BPE merge-loop progress",
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

    if args.progress_interval_bytes <= 0:
        args.progress_interval_bytes = None

    special_tokens = [s.strip() for s in args.special_tokens.split(",") if s.strip()]
    worker_counts = _normalize_worker_counts(args.num_workers)

    experiments_dir = Path(args.experiments_dir)
    experiments_dir.mkdir(parents=True, exist_ok=True)

    experiment_records: list[dict[str, Any]] = []
    final_train_path: Path | None = None
    final_val_path: Path | None = None

    for worker_count in worker_counts:
        final_train_path, final_val_path, experiment_record = _run_experiment(
            worker_count=worker_count,
            args=args,
            special_tokens=special_tokens,
        )
        experiment_records.append(experiment_record)

    _write_json(
        experiments_dir / "summary.json",
        {"experiments": experiment_records},
    )

    if len(worker_counts) == 1 and final_train_path is not None and final_val_path is not None:
        out_train_path = Path(args.out_train_npy)
        out_val_path = Path(args.out_val_npy)
        out_train_path.parent.mkdir(parents=True, exist_ok=True)
        out_val_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(final_train_path, out_train_path)
        shutil.copyfile(final_val_path, out_val_path)

        train_stats = experiment_records[-1]["splits"]["train"]
        val_stats = experiment_records[-1]["splits"]["val"]
        print(f"Saved train: {args.out_train_npy}  {_describe_tokens(train_stats)}")
        print(f"Saved val:   {args.out_val_npy}  {_describe_tokens(val_stats)}")
    else:
        print(
            f"Saved experiment artifacts under {experiments_dir} "
            f"for worker counts: {', '.join(str(count) for count in worker_counts)}"
        )


if __name__ == "__main__":
    main()
