from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
import multiprocessing as mp
from typing import Any, Iterable

from .tokenizer import (
    DEFAULT_PROGRESS_INTERVAL_BYTES,
    DEFAULT_READ_CHUNK_SIZE_BYTES,
    ProgressCallback,
    iter_pretokens_from_file,
)


def _emit_progress(
    callback: ProgressCallback | None,
    **event: Any,
) -> None:
    if callback is not None:
        callback(event)


def _chunked(items: Iterable[str], chunk_size: int) -> Iterable[list[str]]:
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _count_word_freq_batch(
    batch: list[str],
    byte_ids_start: int,
) -> Counter[tuple[int, ...]]:
    word_freq: Counter[tuple[int, ...]] = Counter()
    for token in batch:
        byte_values = token.encode("utf-8", errors="replace")
        word = tuple(byte_ids_start + value for value in byte_values)
        word_freq[word] += 1
    return word_freq


def _count_word_freq_batch_star(
    args: tuple[list[str], int],
) -> Counter[tuple[int, ...]]:
    batch, byte_ids_start = args
    return _count_word_freq_batch(batch, byte_ids_start)


@dataclass
class BPETrainer:
    vocab_size: int
    special_tokens: list[str]
    num_workers: int = 1
    pretoken_batch_size: int = 50_000

    def _build_word_freq(
        self,
        pretokens: Iterable[str],
        byte_ids_start: int,
        num_workers: int,
        pretoken_batch_size: int,
    ) -> Counter[tuple[int, ...]]:
        word_freq: Counter[tuple[int, ...]] = Counter()

        if num_workers == 1:
            for batch in _chunked(pretokens, pretoken_batch_size):
                word_freq.update(_count_word_freq_batch(batch, byte_ids_start))
            return word_freq

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            tasks = (
                (batch, byte_ids_start)
                for batch in _chunked(pretokens, pretoken_batch_size)
            )
            for partial_freq in pool.imap_unordered(
                _count_word_freq_batch_star,
                tasks,
                chunksize=1,
            ):
                word_freq.update(partial_freq)

        return word_freq

    def train(
        self,
        input_path: str,
        num_workers: int | None = None,
        pretoken_batch_size: int | None = None,
        read_chunk_size_bytes: int = DEFAULT_READ_CHUNK_SIZE_BYTES,
        progress_interval_bytes: int | None = DEFAULT_PROGRESS_INTERVAL_BYTES,
        progress_callback: ProgressCallback | None = None,
        merge_progress_every: int = 100,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        worker_count = self.num_workers if num_workers is None else int(num_workers)
        if worker_count < 1:
            raise ValueError("num_workers must be >= 1")

        batch_size = (
            self.pretoken_batch_size
            if pretoken_batch_size is None
            else int(pretoken_batch_size)
        )
        if batch_size < 1:
            raise ValueError("pretoken_batch_size must be >= 1")
        if read_chunk_size_bytes < 1:
            raise ValueError("read_chunk_size_bytes must be >= 1")
        if merge_progress_every < 1:
            raise ValueError("merge_progress_every must be >= 1")

        vocab: dict[int, bytes] = {}
        merges: list[tuple[bytes, bytes]] = []

        next_id = 0
        for special_token in self.special_tokens:
            vocab[next_id] = special_token.encode("utf-8")
            next_id += 1

        byte_ids_start = next_id
        for byte_value in range(256):
            vocab[next_id] = bytes([byte_value])
            next_id += 1

        num_merges = self.vocab_size - len(self.special_tokens) - 256
        if num_merges <= 0:
            return vocab, merges

        word_freq = self._build_word_freq(
            pretokens=iter_pretokens_from_file(
                path=input_path,
                special_tokens=self.special_tokens,
                read_chunk_size_bytes=read_chunk_size_bytes,
                progress_interval_bytes=progress_interval_bytes,
                progress_callback=progress_callback,
                progress_stage="bpe_count",
            ),
            byte_ids_start=byte_ids_start,
            num_workers=worker_count,
            pretoken_batch_size=batch_size,
        )

        _emit_progress(
            progress_callback,
            stage="bpe_count_complete",
            unique_words=len(word_freq),
            input_path=input_path,
        )

        if not word_freq:
            return vocab, merges

        words: list[list[int]] = [list(word) for word in word_freq.keys()]
        freqs: list[int] = [word_freq[tuple(word)] for word in word_freq.keys()]

        def iter_pairs(word: list[int]) -> Iterable[tuple[int, int]]:
            for index in range(len(word) - 1):
                yield word[index], word[index + 1]

        def merge_pair_in_word(
            word: list[int],
            left_id: int,
            right_id: int,
            new_id: int,
        ) -> list[int]:
            merged_word: list[int] = []
            index = 0
            while index < len(word):
                if (
                    index + 1 < len(word)
                    and word[index] == left_id
                    and word[index + 1] == right_id
                ):
                    merged_word.append(new_id)
                    index += 2
                else:
                    merged_word.append(word[index])
                    index += 1
            return merged_word

        pair_counts: Counter[tuple[int, int]] = Counter()
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)

        for word_index, word in enumerate(words):
            freq = freqs[word_index]
            if freq <= 0 or len(word) < 2:
                continue
            for pair in iter_pairs(word):
                pair_counts[pair] += freq
                pair_to_words[pair].add(word_index)

        for merge_index in range(num_merges):
            best_pair = None
            best_key = None

            for (left_id, right_id), count in pair_counts.items():
                if count <= 0:
                    continue
                key = (count, vocab[left_id], vocab[right_id])
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = (left_id, right_id)

            if best_pair is None:
                break

            left_id, right_id = best_pair
            new_token_bytes = vocab[left_id] + vocab[right_id]
            new_id = next_id
            vocab[new_id] = new_token_bytes
            next_id += 1
            merges.append((vocab[left_id], vocab[right_id]))

            affected_words = list(pair_to_words.get((left_id, right_id), ()))
            if not affected_words:
                pair_counts[(left_id, right_id)] = 0
            else:
                for word_index in affected_words:
                    old_word = words[word_index]
                    freq = freqs[word_index]
                    if freq <= 0 or len(old_word) < 2:
                        continue

                    old_pairs = list(iter_pairs(old_word))
                    for pair in old_pairs:
                        pair_counts[pair] -= freq
                        indices = pair_to_words.get(pair)
                        if indices is not None:
                            indices.discard(word_index)
                            if not indices:
                                pair_to_words.pop(pair, None)

                    new_word = merge_pair_in_word(old_word, left_id, right_id, new_id)
                    words[word_index] = new_word

                    if len(new_word) >= 2:
                        for pair in iter_pairs(new_word):
                            pair_counts[pair] += freq
                            pair_to_words[pair].add(word_index)

                pair_counts[(left_id, right_id)] = 0
                pair_to_words.pop((left_id, right_id), None)

            if (
                (merge_index + 1) % merge_progress_every == 0
                or (merge_index + 1) == num_merges
            ):
                _emit_progress(
                    progress_callback,
                    stage="bpe_merge",
                    completed_merges=merge_index + 1,
                    total_merges=num_merges,
                    vocab_size=len(vocab),
                )

        return vocab, merges
