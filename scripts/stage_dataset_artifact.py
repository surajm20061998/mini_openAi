from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.wandb_utils import (
    configure_wandb_environment,
    download_artifact,
    resolve_dataset_files,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a W&B dataset artifact to local disk.")
    p.add_argument("--artifact", type=str, required=True, help="Artifact ref like entity/project/name:alias")
    p.add_argument("--out_dir", type=str, default="data/wandb_dataset")
    p.add_argument("--train_name", type=str, default="train_tokens_full_w8.npy")
    p.add_argument("--val_name", type=str, default="val_tokens_full_w8.npy")
    p.add_argument("--vocab_name", type=str, default="vocab.json")
    p.add_argument("--merges_name", type=str, default="merges.json")
    p.add_argument("--manifest_path", type=str, default=None, help="Optional JSON manifest path to write")
    p.add_argument("--scratch_dir", type=str, default=None, help="Optional scratch root for W&B dirs/cache")
    p.add_argument("--wandb_dir", type=str, default=None)
    p.add_argument("--wandb_cache_dir", type=str, default=None)
    p.add_argument("--wandb_artifact_dir", type=str, default=None)
    p.add_argument("--wandb_data_dir", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configure_wandb_environment(
        scratch_dir=args.scratch_dir,
        wandb_dir=args.wandb_dir,
        wandb_cache_dir=args.wandb_cache_dir,
        wandb_artifact_dir=args.wandb_artifact_dir,
        wandb_data_dir=args.wandb_data_dir,
    )

    artifact, download_dir = download_artifact(
        artifact_ref=args.artifact,
        root=args.out_dir,
        type_name="dataset",
        run=None,
    )
    files = resolve_dataset_files(
        download_dir,
        train_name=args.train_name,
        val_name=args.val_name,
        vocab_name=args.vocab_name,
        merges_name=args.merges_name,
    )

    payload = {
        "artifact_ref": args.artifact,
        "artifact_version": getattr(artifact, "version", None),
        "download_dir": str(download_dir),
        **{key: str(value) for key, value in files.items()},
    }
    if args.manifest_path:
        manifest_path = Path(args.manifest_path).expanduser().resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        payload["manifest_path"] = str(manifest_path)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
