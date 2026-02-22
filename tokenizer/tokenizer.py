from __future__ import annotations

import codecs
import json
from itertools import islice
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import regex as re

from .constants import PAT


EncodeUnit = tuple[str, bool]
ProgressCallback = Callable[[dict[str, Any]], None]

DEFAULT_READ_CHUNK_SIZE_BYTES = 4_194_304
DEFAULT_PROGRESS_INTERVAL_BYTES = 134_217_728

_TOKENIZER_WORKER: Tokenizer | None = None


def _emit_progress(
    callback: ProgressCallback | None,
    **event: Any,
) -> None:
    if callback is not None:
        callback(event)


def _build_special_split_re(special_tokens: Sequence[str]) -> re.Pattern[str] | None:
    if not special_tokens:
        return None
    ordered = sorted(special_tokens, key=len, reverse=True)
    return re.compile("(" + "|".join(re.escape(token) for token in ordered) + ")")


def _chunked(items: Iterable[EncodeUnit], chunk_size: int) -> Iterator[list[EncodeUnit]]:
    iterator = iter(items)
    while True:
        chunk = list(islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def _iter_text_chunks_from_file(
    path: str | Path,
    read_chunk_size_bytes: int = DEFAULT_READ_CHUNK_SIZE_BYTES,
    progress_interval_bytes: int | None = DEFAULT_PROGRESS_INTERVAL_BYTES,
    progress_callback: ProgressCallback | None = None,
    progress_stage: str = "file_read",
) -> Iterator[str]:
    if read_chunk_size_bytes < 1:
        raise ValueError("read_chunk_size_bytes must be >= 1")

    file_path = Path(path)
    total_bytes = file_path.stat().st_size
    bytes_processed = 0
    last_report = 0
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    with open(file_path, "rb") as f:
        while True:
            raw = f.read(read_chunk_size_bytes)
            if not raw:
                break

            bytes_processed += len(raw)
            chunk = decoder.decode(raw, final=False)
            if chunk:
                yield chunk

            if (
                progress_interval_bytes is not None
                and (
                    bytes_processed - last_report >= progress_interval_bytes
                    or bytes_processed == total_bytes
                )
            ):
                last_report = bytes_processed
                _emit_progress(
                    progress_callback,
                    stage=progress_stage,
                    bytes_processed=bytes_processed,
                    total_bytes=total_bytes,
                    path=str(file_path),
                )

        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail

    if progress_interval_bytes is None and total_bytes > 0:
        return

    if bytes_processed == total_bytes and bytes_processed != last_report:
        _emit_progress(
            progress_callback,
            stage=progress_stage,
            bytes_processed=bytes_processed,
            total_bytes=total_bytes,
            path=str(file_path),
        )


def _consume_nonspecial_units(
    text: str,
    pretok_re: re.Pattern[str],
    *,
    final_segment: bool,
) -> tuple[list[EncodeUnit], str]:
    emitted: list[EncodeUnit] = []
    if not text:
        return emitted, ""

    carry_start = len(text)
    for match in pretok_re.finditer(text):
        token = match.group(0)
        if not token:
            continue

        if not final_segment and match.end() == len(text):
            carry_start = match.start()
            break

        emitted.append((token, False))
        carry_start = match.end()

    if final_segment:
        return emitted, ""

    return emitted, text[carry_start:]


def _consume_buffer_to_units(
    buffer: str,
    pretok_re: re.Pattern[str],
    special_tokens: Sequence[str],
    special_split_re: re.Pattern[str] | None,
    *,
    final_buffer: bool,
) -> tuple[list[EncodeUnit], str]:
    emitted: list[EncodeUnit] = []
    if not buffer:
        return emitted, ""

    if special_split_re is None:
        return _consume_nonspecial_units(
            buffer,
            pretok_re,
            final_segment=final_buffer,
        )

    if final_buffer:
        cursor = 0
        for match in special_split_re.finditer(buffer):
            segment_units, _ = _consume_nonspecial_units(
                buffer[cursor : match.start()],
                pretok_re,
                final_segment=True,
            )
            emitted.extend(segment_units)
            emitted.append((match.group(0), True))
            cursor = match.end()

        segment_units, _ = _consume_nonspecial_units(
            buffer[cursor:],
            pretok_re,
            final_segment=True,
        )
        emitted.extend(segment_units)
        return emitted, ""

    max_special_length = max(len(token) for token in special_tokens)
    safe_special_start = max(0, len(buffer) - max_special_length + 1)

    cursor = 0
    for match in special_split_re.finditer(buffer):
        if match.start() >= safe_special_start:
            break

        segment_units, _ = _consume_nonspecial_units(
            buffer[cursor : match.start()],
            pretok_re,
            final_segment=True,
        )
        emitted.extend(segment_units)
        emitted.append((match.group(0), True))
        cursor = match.end()

    tail = buffer[cursor:]
    safe_plain_end = max(0, safe_special_start - cursor)
    safe_plain_prefix = tail[:safe_plain_end]

    prefix_units, prefix_leftover = _consume_nonspecial_units(
        safe_plain_prefix,
        pretok_re,
        final_segment=False,
    )
    emitted.extend(prefix_units)

    leftover = prefix_leftover + tail[safe_plain_end:]
    return emitted, leftover


def iter_encode_units_from_chunks(
    chunks: Iterable[str],
    special_tokens: Sequence[str],
    pretok_re: re.Pattern[str] | None = None,
    special_split_re: re.Pattern[str] | None = None,
) -> Iterator[EncodeUnit]:
    compiled_pretok_re = pretok_re or re.compile(PAT)
    compiled_special_split_re = (
        special_split_re
        if special_split_re is not None
        else _build_special_split_re(special_tokens)
    )

    buffer = ""
    for chunk in chunks:
        if not chunk:
            continue

        buffer += chunk
        emitted, buffer = _consume_buffer_to_units(
            buffer,
            compiled_pretok_re,
            special_tokens,
            compiled_special_split_re,
            final_buffer=False,
        )
        yield from emitted

    emitted, buffer = _consume_buffer_to_units(
        buffer,
        compiled_pretok_re,
        special_tokens,
        compiled_special_split_re,
        final_buffer=True,
    )
    yield from emitted

    if buffer:
        raise RuntimeError("Tokenizer stream finished with unconsumed buffer")


def iter_encode_units_from_text(
    text: str,
    special_tokens: Sequence[str],
) -> Iterator[EncodeUnit]:
    yield from iter_encode_units_from_chunks((text,), special_tokens)


def iter_encode_units_from_file(
    path: str | Path,
    special_tokens: Sequence[str],
    read_chunk_size_bytes: int = DEFAULT_READ_CHUNK_SIZE_BYTES,
    progress_interval_bytes: int | None = DEFAULT_PROGRESS_INTERVAL_BYTES,
    progress_callback: ProgressCallback | None = None,
    progress_stage: str = "file_read",
) -> Iterator[EncodeUnit]:
    yield from iter_encode_units_from_chunks(
        _iter_text_chunks_from_file(
            path=path,
            read_chunk_size_bytes=read_chunk_size_bytes,
            progress_interval_bytes=progress_interval_bytes,
            progress_callback=progress_callback,
            progress_stage=progress_stage,
        ),
        special_tokens=special_tokens,
    )


def iter_pretokens_from_text(
    text: str,
    special_tokens: Sequence[str],
) -> Iterator[str]:
    for unit_text, is_special in iter_encode_units_from_text(text, special_tokens):
        if not is_special:
            yield unit_text


def iter_pretokens_from_file(
    path: str | Path,
    special_tokens: Sequence[str],
    read_chunk_size_bytes: int = DEFAULT_READ_CHUNK_SIZE_BYTES,
    progress_interval_bytes: int | None = DEFAULT_PROGRESS_INTERVAL_BYTES,
    progress_callback: ProgressCallback | None = None,
    progress_stage: str = "file_read",
) -> Iterator[str]:
    for unit_text, is_special in iter_encode_units_from_file(
        path=path,
        special_tokens=special_tokens,
        read_chunk_size_bytes=read_chunk_size_bytes,
        progress_interval_bytes=progress_interval_bytes,
        progress_callback=progress_callback,
        progress_stage=progress_stage,
    ):
        if not is_special:
            yield unit_text


def _init_tokenizer_worker(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    global _TOKENIZER_WORKER
    _TOKENIZER_WORKER = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
    )


def _encode_units_batch(batch: list[EncodeUnit]) -> list[tuple[str, tuple[int, ...]]]:
    if _TOKENIZER_WORKER is None:
        raise RuntimeError("Tokenizer worker state was not initialized")
    return [
        (unit_text, _TOKENIZER_WORKER._encode_unit(unit_text, is_special))
        for unit_text, is_special in batch
    ]


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
        self._special_split_re = _build_special_split_re(self.special_tokens)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
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

            def to_bytes(x: list[int] | str) -> bytes:
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

    def to_serializable_vocab(self) -> dict[str, list[int]]:
        return {str(idx): list(token_bytes) for idx, token_bytes in self.vocab.items()}

    def to_serializable_merges(self) -> list[list[list[int]]]:
        return [[list(left), list(right)] for left, right in self.merges]

    def _merge_one_best_rank(self, tokens: list[bytes]) -> tuple[bool, list[bytes]]:
        best_i = None
        best_rank = None

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            rank = self.merge_rank.get(pair)
            if rank is not None and (best_rank is None or rank < best_rank):
                best_rank = rank
                best_i = i

        if best_i is None:
            return False, tokens

        merged = tokens[best_i] + tokens[best_i + 1]
        new_tokens = tokens[:best_i] + [merged] + tokens[best_i + 2 :]
        return True, new_tokens

    def _bpe_merge(self, tokens: list[bytes]) -> list[bytes]:
        while True:
            did_merge, tokens = self._merge_one_best_rank(tokens)
            if not did_merge:
                break
        return tokens

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        byte_tokens = self._bytes_to_byte_tokens(pretoken.encode("utf-8"))
        merged_tokens = self._bpe_merge(byte_tokens)

        ids: list[int] = []
        for token_bytes in merged_tokens:
            token_id = self.bytes_to_id.get(token_bytes)
            if token_id is None:
                raise KeyError(f"Token bytes not found in vocab: {token_bytes!r}")
            ids.append(token_id)
        return tuple(ids)

    def _encode_unit(self, unit_text: str, is_special: bool) -> tuple[int, ...]:
        if is_special:
            return (self.bytes_to_id[unit_text.encode("utf-8")],)
        return self._encode_pretoken(unit_text)

    def _iter_encode_units(self, text: str) -> Iterator[EncodeUnit]:
        yield from iter_encode_units_from_chunks(
            (text,),
            self.special_tokens,
            pretok_re=self._pretok_re,
            special_split_re=self._special_split_re,
        )

    def _iter_token_sequences_from_units(
        self,
        units: Iterable[EncodeUnit],
        num_workers: int = 1,
        batch_size: int = 4_096,
    ) -> Iterator[tuple[str, tuple[int, ...]]]:
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        if num_workers == 1:
            for unit_text, is_special in units:
                yield unit_text, self._encode_unit(unit_text, is_special)
            return

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_workers,
            initializer=_init_tokenizer_worker,
            initargs=(self.vocab, self.merges, self.special_tokens),
        ) as pool:
            for encoded_batch in pool.imap(
                _encode_units_batch,
                _chunked(units, batch_size),
                chunksize=1,
            ):
                for item in encoded_batch:
                    yield item

    def iter_token_sequences(
        self,
        text: str,
        num_workers: int = 1,
        batch_size: int = 4_096,
    ) -> Iterator[tuple[str, tuple[int, ...]]]:
        yield from self._iter_token_sequences_from_units(
            self._iter_encode_units(text),
            num_workers=num_workers,
            batch_size=batch_size,
        )

    def iter_token_sequences_from_file(
        self,
        path: str | Path,
        num_workers: int = 1,
        batch_size: int = 4_096,
        read_chunk_size_bytes: int = DEFAULT_READ_CHUNK_SIZE_BYTES,
        progress_interval_bytes: int | None = DEFAULT_PROGRESS_INTERVAL_BYTES,
        progress_callback: ProgressCallback | None = None,
        progress_stage: str = "tokenization",
    ) -> Iterator[tuple[str, tuple[int, ...]]]:
        yield from self._iter_token_sequences_from_units(
            iter_encode_units_from_file(
                path=path,
                special_tokens=self.special_tokens,
                read_chunk_size_bytes=read_chunk_size_bytes,
                progress_interval_bytes=progress_interval_bytes,
                progress_callback=progress_callback,
                progress_stage=progress_stage,
            ),
            num_workers=num_workers,
            batch_size=batch_size,
        )

    def _encode_nonspecial_segment(self, text: str) -> list[int]:
        out_ids: list[int] = []
        for match in self._pretok_re.finditer(text):
            pretoken = match.group(0)
            out_ids.extend(self._encode_pretoken(pretoken))
        return out_ids

    def encode(
        self,
        text: str,
        num_workers: int = 1,
        batch_size: int = 4_096,
    ) -> list[int]:
        ids: list[int] = []
        for _, sequence in self.iter_token_sequences(
            text,
            num_workers=num_workers,
            batch_size=batch_size,
        ):
            ids.extend(sequence)
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_string = b"".join(self.vocab[i] for i in ids)
        return byte_string.decode("utf-8", errors="replace")
