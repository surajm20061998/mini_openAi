from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from tokenizer.tokenizer import (
    DEFAULT_PROGRESS_INTERVAL_BYTES,
    DEFAULT_READ_CHUNK_SIZE_BYTES,
    ProgressCallback,
    Tokenizer,
)


DEFAULT_TOKEN_FLUSH_SIZE = 4_194_304


def _build_sequence_payload(
    tokenizer: Tokenizer,
    sequence: tuple[int, ...] | None,
    source_text: str | None,
    count: int = 0,
) -> dict[str, Any] | None:
    if not sequence or source_text is None:
        return None
    return {
        "text": source_text,
        "decoded_text": tokenizer.decode(list(sequence)),
        "token_ids": list(sequence),
        "length": len(sequence),
        "count": count,
    }


def _build_stats(
    tokenizer: Tokenizer,
    pretoken_counter: Counter[str],
    sequence_counter: Counter[tuple[int, ...]],
    sequence_examples: dict[tuple[int, ...], str],
    longest_sequence: tuple[int, ...] | None,
    longest_source_text: str | None,
    sequence_count: int,
    token_count: int,
    original_bytes: int,
    elapsed_seconds: float,
    num_workers: int,
    token_id_min: int | None,
    token_id_max: int | None,
) -> dict[str, Any]:
    total_occurrences = sum(pretoken_counter.values())
    exact_vocab_occurrences = sum(
        count
        for pretoken, count in pretoken_counter.items()
        if pretoken.encode("utf-8") in tokenizer.bytes_to_id
    )
    unique_pretokens = len(pretoken_counter)
    exact_vocab_types = sum(
        1
        for pretoken in pretoken_counter
        if pretoken.encode("utf-8") in tokenizer.bytes_to_id
    )

    vocabulary_coverage = (
        exact_vocab_occurrences / total_occurrences if total_occurrences else 1.0
    )
    oov_rate = 1.0 - vocabulary_coverage
    vocabulary_coverage_by_type = (
        exact_vocab_types / unique_pretokens if unique_pretokens else 1.0
    )

    most_common_sequence: tuple[int, ...] | None = None
    most_common_count = 0
    if sequence_counter:
        most_common_sequence, most_common_count = max(
            sequence_counter.items(),
            key=lambda item: (item[1], len(item[0]), item[0]),
        )

    return {
        "num_workers": num_workers,
        "original_bytes": original_bytes,
        "num_sequences": sequence_count,
        "num_tokens": token_count,
        "unique_sequences": len(sequence_counter),
        "unique_pretokens": unique_pretokens,
        "normalized_sequence_length": (
            float(token_count) / sequence_count if sequence_count else 0.0
        ),
        "vocabulary_coverage": vocabulary_coverage,
        "oov_rate": oov_rate,
        "vocabulary_coverage_by_type": vocabulary_coverage_by_type,
        "compression_ratio": (
            original_bytes / float(token_count) if token_count else 0.0
        ),
        "token_id_min": token_id_min,
        "token_id_max": token_id_max,
        "tokenization_speed": {
            "elapsed_seconds": elapsed_seconds,
            "tokens_per_second": (
                float(token_count) / elapsed_seconds
                if elapsed_seconds > 0
                else float("inf")
            ),
            "sequences_per_second": (
                float(sequence_count) / elapsed_seconds
                if elapsed_seconds > 0
                else float("inf")
            ),
            "bytes_per_second": (
                float(original_bytes) / elapsed_seconds
                if elapsed_seconds > 0
                else float("inf")
            ),
        },
        "longest_sequence": _build_sequence_payload(
            tokenizer=tokenizer,
            sequence=longest_sequence,
            source_text=longest_source_text,
            count=sequence_counter.get(longest_sequence, 0),
        ),
        "most_occuring_sequence": _build_sequence_payload(
            tokenizer=tokenizer,
            sequence=most_common_sequence,
            source_text=sequence_examples.get(most_common_sequence)
            if most_common_sequence is not None
            else None,
            count=most_common_count,
        ),
    }


def collect_tokenization_stats(
    tokenizer: Tokenizer,
    text: str,
    num_workers: int = 1,
    batch_size: int = 4_096,
) -> tuple[np.ndarray, dict[str, Any]]:
    start = time.perf_counter()

    token_ids: list[int] = []
    sequence_counter: Counter[tuple[int, ...]] = Counter()
    sequence_examples: dict[tuple[int, ...], str] = {}
    pretoken_counter: Counter[str] = Counter()

    longest_sequence: tuple[int, ...] | None = None
    longest_source_text: str | None = None
    sequence_count = 0
    token_id_min: int | None = None
    token_id_max: int | None = None

    for source_text, sequence in tokenizer.iter_token_sequences(
        text,
        num_workers=num_workers,
        batch_size=batch_size,
    ):
        if not sequence:
            continue

        sequence_count += 1
        token_ids.extend(sequence)
        sequence_counter[sequence] += 1
        sequence_examples.setdefault(sequence, source_text)
        pretoken_counter[source_text] += 1

        sequence_min = min(sequence)
        sequence_max = max(sequence)
        token_id_min = sequence_min if token_id_min is None else min(token_id_min, sequence_min)
        token_id_max = sequence_max if token_id_max is None else max(token_id_max, sequence_max)

        if longest_sequence is None or (len(sequence), sequence) > (
            len(longest_sequence),
            longest_sequence,
        ):
            longest_sequence = sequence
            longest_source_text = source_text

    elapsed_seconds = time.perf_counter() - start
    token_array = np.asarray(token_ids, dtype=np.int32)
    original_bytes = len(text.encode("utf-8", errors="replace"))

    stats = _build_stats(
        tokenizer=tokenizer,
        pretoken_counter=pretoken_counter,
        sequence_counter=sequence_counter,
        sequence_examples=sequence_examples,
        longest_sequence=longest_sequence,
        longest_source_text=longest_source_text,
        sequence_count=sequence_count,
        token_count=int(token_array.size),
        original_bytes=original_bytes,
        elapsed_seconds=elapsed_seconds,
        num_workers=num_workers,
        token_id_min=token_id_min,
        token_id_max=token_id_max,
    )

    return token_array, stats


def _write_raw_tokens_to_npy(
    raw_path: Path,
    output_path: Path,
    token_count: int,
    copy_chunk_tokens: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_memmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.int32,
        shape=(token_count,),
    )

    try:
        if token_count == 0:
            output_memmap.flush()
            return

        raw_memmap = np.memmap(
            raw_path,
            dtype=np.int32,
            mode="r",
            shape=(token_count,),
        )
        try:
            for start in range(0, token_count, copy_chunk_tokens):
                end = min(token_count, start + copy_chunk_tokens)
                output_memmap[start:end] = raw_memmap[start:end]
            output_memmap.flush()
        finally:
            del raw_memmap
    finally:
        del output_memmap


def tokenize_file_to_npy_and_collect_stats(
    tokenizer: Tokenizer,
    input_path: str | Path,
    output_path: str | Path,
    num_workers: int = 1,
    batch_size: int = 4_096,
    read_chunk_size_bytes: int = DEFAULT_READ_CHUNK_SIZE_BYTES,
    progress_interval_bytes: int | None = DEFAULT_PROGRESS_INTERVAL_BYTES,
    progress_callback: ProgressCallback | None = None,
    progress_stage: str = "tokenization",
    token_flush_size: int = DEFAULT_TOKEN_FLUSH_SIZE,
    copy_chunk_tokens: int = DEFAULT_TOKEN_FLUSH_SIZE,
) -> dict[str, Any]:
    if token_flush_size < 1:
        raise ValueError("token_flush_size must be >= 1")
    if copy_chunk_tokens < 1:
        raise ValueError("copy_chunk_tokens must be >= 1")

    input_path_obj = Path(input_path)
    output_path_obj = Path(output_path)
    raw_path = output_path_obj.with_name(output_path_obj.name + ".raw.tmp")
    original_bytes = input_path_obj.stat().st_size

    start = time.perf_counter()

    sequence_counter: Counter[tuple[int, ...]] = Counter()
    sequence_examples: dict[tuple[int, ...], str] = {}
    pretoken_counter: Counter[str] = Counter()
    longest_sequence: tuple[int, ...] | None = None
    longest_source_text: str | None = None
    sequence_count = 0
    token_count = 0
    token_id_min: int | None = None
    token_id_max: int | None = None
    token_buffer: list[int] = []

    try:
        with open(raw_path, "wb") as raw_fp:
            for source_text, sequence in tokenizer.iter_token_sequences_from_file(
                path=input_path_obj,
                num_workers=num_workers,
                batch_size=batch_size,
                read_chunk_size_bytes=read_chunk_size_bytes,
                progress_interval_bytes=progress_interval_bytes,
                progress_callback=progress_callback,
                progress_stage=progress_stage,
            ):
                if not sequence:
                    continue

                sequence_count += 1
                token_count += len(sequence)
                sequence_counter[sequence] += 1
                sequence_examples.setdefault(sequence, source_text)
                pretoken_counter[source_text] += 1
                token_buffer.extend(sequence)

                sequence_min = min(sequence)
                sequence_max = max(sequence)
                token_id_min = sequence_min if token_id_min is None else min(token_id_min, sequence_min)
                token_id_max = sequence_max if token_id_max is None else max(token_id_max, sequence_max)

                if longest_sequence is None or (len(sequence), sequence) > (
                    len(longest_sequence),
                    longest_sequence,
                ):
                    longest_sequence = sequence
                    longest_source_text = source_text

                if len(token_buffer) >= token_flush_size:
                    np.asarray(token_buffer, dtype=np.int32).tofile(raw_fp)
                    token_buffer.clear()

            if token_buffer:
                np.asarray(token_buffer, dtype=np.int32).tofile(raw_fp)
                token_buffer.clear()

        _write_raw_tokens_to_npy(
            raw_path=raw_path,
            output_path=output_path_obj,
            token_count=token_count,
            copy_chunk_tokens=copy_chunk_tokens,
        )
    finally:
        if raw_path.exists():
            raw_path.unlink()

    elapsed_seconds = time.perf_counter() - start

    return _build_stats(
        tokenizer=tokenizer,
        pretoken_counter=pretoken_counter,
        sequence_counter=sequence_counter,
        sequence_examples=sequence_examples,
        longest_sequence=longest_sequence,
        longest_source_text=longest_source_text,
        sequence_count=sequence_count,
        token_count=token_count,
        original_bytes=original_bytes,
        elapsed_seconds=elapsed_seconds,
        num_workers=num_workers,
        token_id_min=token_id_min,
        token_id_max=token_id_max,
    )
