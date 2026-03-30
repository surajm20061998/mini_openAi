from __future__ import annotations

import json
from typing import Iterable, Iterator

import regex as re

from .constants import PAT


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)
        self.special_tokens: list[str] = list(special_tokens or [])
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        next_id = (max(self.vocab.keys()) + 1) if self.vocab else 0
        for s in self.special_tokens:
            bt = s.encode("utf-8")
            if bt not in self.bytes_to_id:
                self.vocab[next_id] = bt
                self.bytes_to_id[bt] = next_id
                next_id += 1

        self.merge_rank: dict[tuple[bytes, bytes], int] = {
            pair: r for r, pair in enumerate(self.merges)
        }

        self._pretok_re = re.compile(PAT)

        if self.special_tokens:
            special_pat = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            self._special_split_re = re.compile(special_pat)
        else:
            self._special_split_re = None

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        Loads vocab + merges from disk. This assumes JSON serialization.
        Since different students serialize differently, this is implemented to be tolerant.

        Recommended formats:
          vocab.json: { "0": [104], "1": [101], ... }  # list of ints = bytes
            OR
          vocab.json: { "0": "68656c6c6f" }  # hex string
        merges.json: [ [[104],[101]], [[...],[...]], ... ]
            OR
        merges.json: [ ["68","65"], ... ] in hex strings
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        vocab: dict[int, bytes] = {}
        for k, v in raw_vocab.items():
            idx = int(k)
            if isinstance(v, list):
                vocab[idx] = bytes(v)
            elif isinstance(v, str):
                try:
                    vocab[idx] = bytes.fromhex(v)
                except ValueError:
                    vocab[idx] = v.encode("latin-1")
            else:
                raise TypeError(f"Unsupported vocab value type: {type(v)}")

        with open(merges_filepath, "r", encoding="utf-8") as f:
            raw_merges = json.load(f)

        merges: list[tuple[bytes, bytes]] = []
        for pair in raw_merges:
            if not (isinstance(pair, list) or isinstance(pair, tuple)) or len(pair) != 2:
                raise ValueError("Each merge must be a 2-item list/tuple")
            a, b = pair

            def to_bytes(x) -> bytes:
                if isinstance(x, list):
                    return bytes(x)
                if isinstance(x, str):
                    try:
                        return bytes.fromhex(x)
                    except ValueError:
                        return x.encode("latin-1")
                raise TypeError(f"Unsupported merge item type: {type(x)}")

            merges.append((to_bytes(a), to_bytes(b)))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @staticmethod
    def _bytes_to_byte_tokens(b: bytes) -> list[bytes]:
        return [bytes([x]) for x in b]

    def _merge_one_best_rank(self, tokens: list[bytes]) -> tuple[bool, list[bytes]]:
        """
        Find the adjacent pair with the smallest merge rank and merge it once.
        Returns (did_merge, new_tokens).
        """
        best_i = None
        best_rank = None

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            r = self.merge_rank.get(pair)
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_i = i

        if best_i is None:
            return False, tokens

        merged = tokens[best_i] + tokens[best_i + 1]
        new_tokens = tokens[:best_i] + [merged] + tokens[best_i + 2 :]
        return True, new_tokens

    def _bpe_merge(self, tokens: list[bytes]) -> list[bytes]:
        """
        Apply BPE merges within ONE pre-token (no crossing pre-token boundaries).
        We repeatedly merge the best-ranked applicable pair until none apply.
        """
        while True:
            did_merge, tokens = self._merge_one_best_rank(tokens)
            if not did_merge:
                break
        return tokens

    def _encode_nonspecial_segment(self, text: str) -> list[int]:
        """
        Encode a string segment that contains NO special tokens.
        Steps: pretokenize -> bytes -> BPE merge -> map to ids.
        """
        out_ids: list[int] = []

        for m in self._pretok_re.finditer(text):
            pre = m.group(0)  # keep leading space
            byte_tokens = self._bytes_to_byte_tokens(pre.encode("utf-8"))
            merged_tokens = self._bpe_merge(byte_tokens)

            for t in merged_tokens:
                tid = self.bytes_to_id.get(t)
                if tid is None:
                    # In a correct setup this shouldn't happen (merged tokens should be in vocab),
                    # but be explicit to avoid silent wrong results.
                    raise KeyError(f"Token bytes not found in vocab: {t!r}")
                out_ids.append(tid)

        return out_ids

    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        Special tokens (if provided) are preserved as single tokens.
        """
        if not self.special_tokens:
            return self._encode_nonspecial_segment(text)

        assert self._special_split_re is not None
        parts = self._special_split_re.split(text) 

        ids: list[int] = []
        for part in parts:
            if part == "" or part is None:
                continue
            if part in self.special_tokens:
                bt = part.encode("utf-8")
                ids.append(self.bytes_to_id[bt])
            else:
                ids.extend(self._encode_nonspecial_segment(part))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Memory-efficient encoding: yields IDs lazily.
        NOTE: This intentionally does NOT allow tokens to cross chunk boundaries.
        """
        for chunk in iterable:
            for tid in self.encode(chunk):
                yield tid

    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back to a Unicode string.
        Invalid UTF-8 byte sequences are replaced with U+FFFD.
        """
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

        