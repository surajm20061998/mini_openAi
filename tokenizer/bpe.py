from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import regex as re
from .constants import PAT

@dataclass
class BPETrainer:
    vocab_size: int
    special_tokens: list[str]

    def train(self, input_path: str) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
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

        pretok_re = re.compile(PAT)

        split_re = None
        if self.special_tokens:
            split_re = re.compile("|".join(re.escape(s) for s in self.special_tokens))
        word_freq: Counter[Tuple[int, ...]] = Counter()

        with open(input_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        segments = split_re.split(text) if split_re is not None else [text]

        for seg in segments:
            if not seg:
                continue
            for m in pretok_re.finditer(seg):
                tok = m.group(0)
                b = tok.encode("utf-8", errors="replace")
                word = tuple(byte_ids_start + x for x in b)
                word_freq[word] += 1

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

        