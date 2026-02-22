from __future__ import annotations

from collections import Counter
from typing import Any
import time

import numpy as np

from tokenizer.tokenizer import Tokenizer


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

        if longest_sequence is None or (len(sequence), sequence) > (
            len(longest_sequence),
            longest_sequence,
        ):
            longest_sequence = sequence
            longest_source_text = source_text

    elapsed = time.perf_counter() - start
    token_array = np.asarray(token_ids, dtype=np.int32)

    original_bytes = len(text.encode("utf-8", errors="replace"))
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

    stats: dict[str, Any] = {
        "num_workers": num_workers,
        "original_bytes": original_bytes,
        "num_sequences": sequence_count,
        "num_tokens": int(token_array.size),
        "unique_sequences": len(sequence_counter),
        "unique_pretokens": unique_pretokens,
        "normalized_sequence_length": (
            float(token_array.size) / sequence_count if sequence_count else 0.0
        ),
        "vocabulary_coverage": vocabulary_coverage,
        "oov_rate": oov_rate,
        "vocabulary_coverage_by_type": vocabulary_coverage_by_type,
        "compression_ratio": (
            original_bytes / float(token_array.size) if token_array.size else 0.0
        ),
        "tokenization_speed": {
            "elapsed_seconds": elapsed,
            "tokens_per_second": (
                float(token_array.size) / elapsed if elapsed > 0 else float("inf")
            ),
            "sequences_per_second": (
                float(sequence_count) / elapsed if elapsed > 0 else float("inf")
            ),
            "bytes_per_second": (
                float(original_bytes) / elapsed if elapsed > 0 else float("inf")
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

    return token_array, stats
