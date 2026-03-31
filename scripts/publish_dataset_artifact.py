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
    parse_csv_values,
    publish_dataset_artifact,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish token files and tokenizer metadata as a W&B dataset artifact.")
    p.add_argument("--project", type=str, required=True, help="W&B project name")
    p.add_argument("--entity", type=str, default=None, help="W&B entity/team name")
    p.add_argument("--artifact_name", type=str, required=True, help="Dataset artifact collection name")
    p.add_argument("--train_tokens_path", type=str, required=True)
    p.add_argument("--val_tokens_path", type=str, required=True)
    p.add_argument("--vocab_path", type=str, required=True)
    p.add_argument("--merges_path", type=str, required=True)
    p.add_argument("--experiment_path", type=str, default=None)
    p.add_argument("--aliases", type=str, nargs="+", default=["latest"], help="Artifact aliases, e.g. latest prod")
    p.add_argument("--mode", type=str, default="online", help="online | offline | disabled")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--description", type=str, default=None)
    p.add_argument("--ttl_days", type=int, default=None, help="Optional TTL in days for W&B-hosted artifacts")
    p.add_argument("--tags", type=str, default=None, help="Comma-separated W&B tags")
    p.add_argument("--metadata_json_path", type=str, default=None, help="Optional JSON file merged into artifact metadata")
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

    extra_metadata = None
    if args.metadata_json_path:
        extra_metadata = json.loads(Path(args.metadata_json_path).read_text(encoding="utf-8"))

    artifact_ref = publish_dataset_artifact(
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        artifact_name=args.artifact_name,
        run_name=args.run_name,
        aliases=args.aliases,
        train_tokens_path=args.train_tokens_path,
        val_tokens_path=args.val_tokens_path,
        vocab_path=args.vocab_path,
        merges_path=args.merges_path,
        experiment_path=args.experiment_path,
        description=args.description,
        ttl_days=args.ttl_days,
        tags=parse_csv_values(args.tags),
        extra_metadata=extra_metadata,
    )
    print(f"[done] published dataset artifact: {artifact_ref}")


if __name__ == "__main__":
    main()
