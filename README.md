### Download data
Download the TinyStories data:

``` sh
mkdir -p data
mkdir -p experiments
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

cd ..
```

### Create smaller datasets for testing

```sh
python3 -c "from pathlib import Path; text=Path('data/TinyStoriesV2-GPT4-train.txt').read_text(encoding='utf-8', errors='ignore').split(); Path('data/train_500000w.txt').write_text(' '.join(text[:500000]), encoding='utf-8')"

python3 -c "from pathlib import Path; text=Path('data/TinyStoriesV2-GPT4-valid.txt').read_text(encoding='utf-8', errors='ignore').split(); Path('data/valid_500000w.txt').write_text(' '.join(text[:500000]), encoding='utf-8')"
```
### Running Tokenization Experiments
```sh

python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 1 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w1.npy \
  --out_val_npy data/val_tokens_500000w_w1.npy


python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 2 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w2.npy \
  --out_val_npy data/val_tokens_500000w_w2.npy


python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 4 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w4.npy \
  --out_val_npy data/val_tokens_500000w_w4.npy


  python3 scripts/make_token_npy.py \
  --train_txt data/train_500000w.txt \
  --val_txt data/valid_500000w.txt \
  --vocab_size 512 \
  --num_workers 8 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_500000w_w8.npy \
  --out_val_npy data/val_tokens_500000w_w8.npy
```

### Running Tokenization Experiments on Full Dataset

```sh
python3 scripts/make_token_npy.py \
  --train_txt data/TinyStoriesV2-GPT4-train.txt \
  --val_txt data/TinyStoriesV2-GPT4-valid.txt \
  --vocab_size 512 \
  --num_workers 8 \
  --experiments_dir experiments \
  --out_train_npy data/train_tokens_full_w8.npy \
  --out_val_npy data/val_tokens_full_w8.npy
```

### Experiment Records are Saved in /experiments

### Download Token Artifacts From Google Drive

```sh
python3 scripts/download_token_artifacts.py \
  --train "GOOGLE_DRIVE_TRAIN_URL_OR_FILE_ID" \
  --val "GOOGLE_DRIVE_VAL_URL_OR_FILE_ID" \
  --vocab "GOOGLE_DRIVE_VOCAB_URL_OR_FILE_ID" \
  --merges "GOOGLE_DRIVE_MERGES_URL_OR_FILE_ID"
```

This creates `data/` if needed and saves the downloaded files there. The Google Drive files must be shared as `Anyone with the link`.

### Publish Token Files To W&B Once

```sh
python3 scripts/publish_dataset_artifact.py \
  --project YOUR_WANDB_PROJECT \
  --entity YOUR_WANDB_ENTITY \
  --artifact_name tinystories-bpe512-w8 \
  --train_tokens_path data/train_tokens_full_w8.npy \
  --val_tokens_path data/val_tokens_full_w8.npy \
  --vocab_path experiments/numWorkers_8/vocab.json \
  --merges_path experiments/numWorkers_8/merges.json \
  --experiment_path experiments/numWorkers_8/experiment.json \
  --aliases latest
```

### Stage A W&B Dataset Artifact Locally

```sh
python3 scripts/stage_dataset_artifact.py \
  --artifact YOUR_WANDB_ENTITY/YOUR_WANDB_PROJECT/tinystories-bpe512-w8:latest \
  --out_dir data/wandb_dataset \
  --manifest_path data/wandb_dataset/manifest.json
```

### Train With W&B Checkpoint Artifacts

```sh
python3 scripts/train_lm.py \
  --train_tokens_path data/train_tokens_full_w8.npy \
  --val_tokens_path data/val_tokens_full_w8.npy \
  --vocab_json_path data/vocab.json \
  --wandb_project trainLLMFromCratch \
  --wandb_entity surajm20061998-new-york-university \
  --wandb_mode online \
  --checkpoint_artifact_name tinystories-transformer-checkpoints \
  --checkpoint_keep_milestone_every 5000 \
  --scratch_dir /tmp/mini_openai_run \
  --context_length 256 \
  --batch_size 32 \
  --d_model 512 \
  --num_layers 8 \
  --num_heads 8 \
  --d_ff 1024 \
  --max_iters 40000 \
  --save_every 500 \
  --eval_every 500 \
  --device auto \
  --dtype float32
```

To resume from W&B:

```sh
python3 scripts/train_lm.py \
  --dataset_artifact surajm20061998-new-york-university/trainLLMFromCratch/tinystories-bpe512-w8:latest \
  --resume_artifact surajm20061998-new-york-university/trainLLMFromCratch/tinystories-transformer-checkpoints:latest \
  --resume \
  --wandb_project trainLLMFromCratch \
  --wandb_entity surajm20061998-new-york-university \
  --wandb_mode online \
  --checkpoint_artifact_name tinystories-transformer-checkpoints \
  --scratch_dir /tmp/mini_openai_run
```

### Run A Compute-Optimal Scaling Sweep

This sweep runs the following model sizes:

- Small: `d_model=256`, `num_layers=4`, `num_heads=4`, `d_ff=768`
- Medium: `d_model=384`, `num_layers=6`, `num_heads=6`, `d_ff=1024`
- Large: `d_model=512`, `num_layers=8`, `num_heads=8`, `d_ff=1536`

and the following token budgets:

- `25M`
- `50M`
- `100M`
- `200M`

Each run is recorded under `sweep_experiments/`, and the trainer now logs:

- `val/loss`
- `val/ppl`
- `train/tokens_seen`
- `perf/wall_clock_seconds`
- `perf/tokens_per_second`
- `model/parameter_count`
- `perf/flops_proxy`

Run the sweep with local token files:

```sh
WANDB_PROJECT=trainLLMFromCratch \
WANDB_ENTITY=surajm20061998-new-york-university \
DEVICE=auto \
DTYPE=float32 \
bash scripts/run_compute_optimal_scaling_sweep.sh
```

Or run the same sweep against a W&B dataset artifact:

```sh
WANDB_PROJECT=trainLLMFromCratch \
WANDB_ENTITY=surajm20061998-new-york-university \
DATASET_ARTIFACT=surajm20061998-new-york-university/trainLLMFromCratch/tinystories-bpe512-w8:latest \
DEVICE=auto \
DTYPE=float32 \
bash scripts/run_compute_optimal_scaling_sweep.sh
```
