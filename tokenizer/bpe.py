from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
import multiprocessing as mp
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import regex as re

from .constants import PAT


def _build_special_split_re(special_tokens: Sequence[str]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    ordered = sorted(special_tokens, key=len, reverse=True)
    return re.compile("|".join(re.escape(token) for token in ordered))


def _iter_training_pretokens(text: str, special_tokens: Sequence[str]) -> Iterator[str]:
    pretok_re = re.compile(PAT)
    split_re = _build_special_split_re(special_tokens)
    segments = split_re.split(text) if split_re is not None else [text]

    for segment in segments:
        if not segment:
            continue
        for match in pretok_re.finditer(segment):
            token = match.group(0)
            if token:
                yield token


def _chunked(items: Iterable[str], chunk_size: int) -> Iterator[list[str]]:
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _count_word_freq_batch(
    batch: list[str],
    byte_ids_start: int,
) -> Counter[Tuple[int, ...]]:
    word_freq: Counter[Tuple[int, ...]] = Counter()
    for token in batch:
        byte_values = token.encode("utf-8", errors="replace")
        word = tuple(byte_ids_start + value for value in byte_values)
        word_freq[word] += 1
    return word_freq


def _count_word_freq_batch_star(
    args: tuple[list[str], int],
) -> Counter[Tuple[int, ...]]:
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
        text: str,
        byte_ids_start: int,
        num_workers: int,
        pretoken_batch_size: int,
    ) -> Counter[Tuple[int, ...]]:
        pretokens = _iter_training_pretokens(text, self.special_tokens)
        word_freq: Counter[Tuple[int, ...]] = Counter()

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

        vocab: Dict[int, bytes] = {}
        merges: List[Tuple[bytes, bytes]] = []

        next_id = 0
        for s in self.special_tokens:
            vocab[next_id] = s.encode("utf-8")
            next_id += 1

        byte_ids_start = next_id
        for i in range(256):
            vocab[next_id] = bytes([i])
            next_id += 1

        num_merges = self.vocab_size - len(self.special_tokens) - 256
        if num_merges <= 0:
            return vocab, merges

        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        word_freq = self._build_word_freq(
            text=text,
            byte_ids_start=byte_ids_start,
            num_workers=worker_count,
            pretoken_batch_size=batch_size,
        )

        if not word_freq:
            return vocab, merges

        words: List[List[int]] = [list(w) for w in word_freq.keys()]
        freqs: List[int] = [word_freq[tuple(w)] for w in word_freq.keys()]

        def iter_pairs(w: List[int]):
            for i in range(len(w) - 1):
                yield (w[i], w[i + 1])

        def merge_pair_in_word(w: List[int], a: int, b: int, new_id: int) -> List[int]:
            out: List[int] = []
            i = 0
            n = len(w)
            while i < n:
                if i + 1 < n and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            return out

        pair_counts: Counter[Tuple[int, int]] = Counter()
        pair_to_words: dict[Tuple[int, int], set[int]] = defaultdict(set)

        for wi, w in enumerate(words):
            f = freqs[wi]
            if f <= 0 or len(w) < 2:
                continue
            for p in iter_pairs(w):
                pair_counts[p] += f
                pair_to_words[p].add(wi)

        for _ in range(num_merges):

            best_pair = None
            best_key = None

            for (a, b), c in pair_counts.items():
                if c <= 0:
                    continue
                key = (c, vocab[a], vocab[b])
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = (a, b)

            if best_pair is None:
                break

            a, b = best_pair
            new_bytes = vocab[a] + vocab[b]
            new_id = next_id
            vocab[new_id] = new_bytes
            next_id += 1
            merges.append((vocab[a], vocab[b]))

            affected = list(pair_to_words.get((a, b), ()))
            if not affected:
                pair_counts[(a, b)] = 0
                continue

            for wi in affected:
                old_w = words[wi]
                f = freqs[wi]
                if f <= 0 or len(old_w) < 2:
                    continue

                old_pairs = list(iter_pairs(old_w))
                for p in old_pairs:
                    pair_counts[p] -= f
                    s = pair_to_words.get(p)
                    if s is not None:
                        s.discard(wi)
                        if not s:
                            pair_to_words.pop(p, None)

                new_w = merge_pair_in_word(old_w, a, b, new_id)
                words[wi] = new_w

                if len(new_w) >= 2:
                    for p in iter_pairs(new_w):
                        pair_counts[p] += f
                        pair_to_words[p].add(wi)

            pair_counts[(a, b)] = 0
            pair_to_words.pop((a, b), None)

        return vocab, merges
