from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from tokenizer.bpe import BPETrainer
from tokenizer.tokenizer import Tokenizer


def encode_file(tokenizer: Tokenizer, path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    ids = tokenizer.encode(text)
    arr = np.asarray(ids, dtype=np.int32)
    return arr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train_txt", type=str, required=True)
    p.add_argument("--val_txt", type=str, required=True)
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument(
        "--special_tokens",
        type=str,
        default="<|endoftext|>",
        help="Comma-separated list, default includes <|endoftext|>",
    )
    p.add_argument("--out_train_npy", type=str, default="data/train_tokens.npy")
    p.add_argument("--out_val_npy", type=str, default="data/val_tokens.npy")
    args = p.parse_args()

    special_tokens = [s.strip() for s in args.special_tokens.split(",") if s.strip()]

    trainer = BPETrainer(vocab_size=args.vocab_size, special_tokens=special_tokens)
    vocab, merges = trainer.train(args.train_txt)

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)

    train_arr = encode_file(tokenizer, args.train_txt)
    val_arr = encode_file(tokenizer, args.val_txt)

    Path(args.out_train_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_train_npy, train_arr)
    np.save(args.out_val_npy, val_arr)

    print(f"Saved train: {args.out_train_npy}  shape={train_arr.shape} dtype={train_arr.dtype} "
          f"min={train_arr.min()} max={train_arr.max()}")
    print(f"Saved val:   {args.out_val_npy}  shape={val_arr.shape} dtype={val_arr.dtype} "
          f"min={val_arr.min()} max={val_arr.max()}")


if __name__ == "__main__":
    main()