from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any


DEFAULT_DATASET_FILENAMES = {
    "train_tokens": "train_tokens_full_w8.npy",
    "val_tokens": "val_tokens_full_w8.npy",
    "vocab": "vocab.json",
    "merges": "merges.json",
    "experiment": "experiment.json",
}


def wandb_is_enabled(mode: str | None) -> bool:
    return bool(mode) and mode.lower() != "disabled"


def require_wandb() -> Any:
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases support requires the `wandb` package. "
            "Install it with `pip install wandb` or `pip install -r requirements.txt`."
        ) from exc
    return wandb


def parse_csv_values(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [value.strip() for value in raw.split(",") if value.strip()]


def ensure_artifact_ref_has_alias(artifact_ref: str, default_alias: str = "latest") -> str:
    artifact_ref = artifact_ref.strip()
    leaf = artifact_ref.rsplit("/", 1)[-1]
    if ":" in leaf:
        return artifact_ref
    return f"{artifact_ref}:{default_alias}"


def configure_wandb_environment(
    *,
    scratch_dir: str | None = None,
    wandb_dir: str | None = None,
    wandb_cache_dir: str | None = None,
    wandb_artifact_dir: str | None = None,
    wandb_data_dir: str | None = None,
) -> dict[str, str]:
    scratch_path = Path(scratch_dir).expanduser().resolve() if scratch_dir else None

    if scratch_path is not None:
        wandb_dir = wandb_dir or str(scratch_path / "wandb" / "runs")
        wandb_cache_dir = wandb_cache_dir or str(scratch_path / "wandb" / "cache")
        wandb_artifact_dir = wandb_artifact_dir or str(scratch_path / "wandb" / "artifacts")
        wandb_data_dir = wandb_data_dir or str(scratch_path / "wandb" / "data")

    resolved: dict[str, str] = {}
    for env_name, raw_value in (
        ("WANDB_DIR", wandb_dir),
        ("WANDB_CACHE_DIR", wandb_cache_dir),
        ("WANDB_ARTIFACT_DIR", wandb_artifact_dir),
        ("WANDB_DATA_DIR", wandb_data_dir),
    ):
        if not raw_value:
            continue
        path = Path(raw_value).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(path)
        resolved[env_name] = str(path)
    return resolved


def init_wandb_run(
    *,
    project: str,
    entity: str | None,
    mode: str,
    job_type: str,
    name: str | None,
    group: str | None,
    tags: list[str] | None,
    config: dict[str, Any],
) -> Any:
    wandb = require_wandb()
    return wandb.init(
        project=project,
        entity=entity,
        mode=mode,
        job_type=job_type,
        name=name,
        group=group,
        tags=tags or None,
        config=config,
    )


def infer_vocab_size_from_vocab_json(vocab_path: str | Path) -> int:
    payload = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    if isinstance(payload, (dict, list)):
        return len(payload)
    raise ValueError(f"Unsupported vocab.json format in {vocab_path}")


def publish_dataset_artifact(
    *,
    project: str,
    entity: str | None,
    mode: str,
    artifact_name: str,
    run_name: str | None,
    aliases: list[str],
    train_tokens_path: str | Path,
    val_tokens_path: str | Path,
    vocab_path: str | Path,
    merges_path: str | Path,
    experiment_path: str | Path | None,
    description: str | None,
    ttl_days: int | None,
    tags: list[str] | None,
    extra_metadata: dict[str, Any] | None,
) -> str:
    train_path = Path(train_tokens_path).expanduser().resolve()
    val_path = Path(val_tokens_path).expanduser().resolve()
    vocab_json_path = Path(vocab_path).expanduser().resolve()
    merges_json_path = Path(merges_path).expanduser().resolve()
    experiment_json_path = Path(experiment_path).expanduser().resolve() if experiment_path else None

    for path in (train_path, val_path, vocab_json_path, merges_json_path):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata: dict[str, Any] = {
        "train_tokens_filename": DEFAULT_DATASET_FILENAMES["train_tokens"],
        "train_tokens_size_bytes": train_path.stat().st_size,
        "val_tokens_filename": DEFAULT_DATASET_FILENAMES["val_tokens"],
        "val_tokens_size_bytes": val_path.stat().st_size,
        "vocab_filename": DEFAULT_DATASET_FILENAMES["vocab"],
        "vocab_size": infer_vocab_size_from_vocab_json(vocab_json_path),
        "merges_filename": DEFAULT_DATASET_FILENAMES["merges"],
        "merges_size_bytes": merges_json_path.stat().st_size,
    }
    if experiment_json_path is not None and experiment_json_path.exists():
        metadata["experiment_filename"] = DEFAULT_DATASET_FILENAMES["experiment"]
        metadata["experiment_size_bytes"] = experiment_json_path.stat().st_size
    if extra_metadata:
        metadata.update(extra_metadata)

    run = init_wandb_run(
        project=project,
        entity=entity,
        mode=mode,
        job_type="dataset_publish",
        name=run_name,
        group=None,
        tags=tags,
        config={
            "artifact_name": artifact_name,
            "train_tokens_path": str(train_path),
            "val_tokens_path": str(val_path),
            "vocab_path": str(vocab_json_path),
            "merges_path": str(merges_json_path),
            "experiment_path": str(experiment_json_path) if experiment_json_path else None,
        },
    )
    try:
        wandb = require_wandb()
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=description,
            metadata=metadata,
        )
        if ttl_days is not None and ttl_days > 0:
            artifact.ttl = timedelta(days=ttl_days)

        artifact.add_file(str(train_path), name=DEFAULT_DATASET_FILENAMES["train_tokens"])
        artifact.add_file(str(val_path), name=DEFAULT_DATASET_FILENAMES["val_tokens"])
        artifact.add_file(str(vocab_json_path), name=DEFAULT_DATASET_FILENAMES["vocab"])
        artifact.add_file(str(merges_json_path), name=DEFAULT_DATASET_FILENAMES["merges"])
        if experiment_json_path is not None and experiment_json_path.exists():
            artifact.add_file(str(experiment_json_path), name=DEFAULT_DATASET_FILENAMES["experiment"])

        logged_artifact = run.log_artifact(artifact, aliases=aliases)
        logged_artifact.wait()
        return f"{run.entity}/{run.project}/{artifact_name}:{logged_artifact.version}"
    finally:
        run.finish()


def download_artifact(
    *,
    artifact_ref: str,
    root: str | Path | None,
    type_name: str | None,
    run: Any | None,
    default_alias: str = "latest",
) -> tuple[Any, Path]:
    require_wandb()
    artifact_ref = ensure_artifact_ref_has_alias(artifact_ref, default_alias=default_alias)
    download_root = Path(root).expanduser().resolve() if root else None
    if download_root is not None:
        download_root.mkdir(parents=True, exist_ok=True)

    if run is not None:
        artifact = run.use_artifact(artifact_ref)
    else:
        wandb = require_wandb()
        api = wandb.Api()
        artifact = api.artifact(name=artifact_ref)

    download_path = Path(artifact.download(root=str(download_root) if download_root else None)).resolve()
    return artifact, download_path


def resolve_dataset_files(
    artifact_dir: str | Path,
    *,
    train_name: str = DEFAULT_DATASET_FILENAMES["train_tokens"],
    val_name: str = DEFAULT_DATASET_FILENAMES["val_tokens"],
    vocab_name: str = DEFAULT_DATASET_FILENAMES["vocab"],
    merges_name: str = DEFAULT_DATASET_FILENAMES["merges"],
) -> dict[str, Path]:
    base_dir = Path(artifact_dir).expanduser().resolve()
    files = {
        "train_tokens_path": base_dir / train_name,
        "val_tokens_path": base_dir / val_name,
        "vocab_path": base_dir / vocab_name,
        "merges_path": base_dir / merges_name,
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset artifact download is missing expected files: " + ", ".join(missing)
        )
    return files


def log_checkpoint_artifact(
    *,
    run: Any,
    artifact_name: str,
    checkpoint_path: str | Path,
    aliases: list[str],
    metadata: dict[str, Any],
    ttl_days: int | None,
) -> Any:
    wandb = require_wandb()
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata,
    )
    if ttl_days is not None and ttl_days > 0:
        artifact.ttl = timedelta(days=ttl_days)
    artifact.add_file(str(ckpt_path), name="checkpoint.pt")
    return run.log_artifact(artifact, aliases=aliases)
