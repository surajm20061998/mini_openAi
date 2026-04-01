#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SWEEP_NAME="${SWEEP_NAME:-compute_optimal_scaling_$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-$PROJECT_ROOT/sweep_experiments/$SWEEP_NAME}"
BASE_SCRATCH="${BASE_SCRATCH:-/tmp/mini_openai_sweeps/$SWEEP_NAME}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-compute_optimal_scaling}"

DATASET_ARTIFACT="${DATASET_ARTIFACT:-}"
TRAIN_TOKENS_PATH="${TRAIN_TOKENS_PATH:-$PROJECT_ROOT/data/train_tokens_full_w8.npy}"
VAL_TOKENS_PATH="${VAL_TOKENS_PATH:-$PROJECT_ROOT/data/val_tokens_full_w8.npy}"
VOCAB_JSON_PATH="${VOCAB_JSON_PATH:-$PROJECT_ROOT/data/vocab.json}"

CONTEXT_LENGTH="${CONTEXT_LENGTH:-256}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_LR="${MAX_LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
BETAS1="${BETAS1:-0.9}"
BETAS2="${BETAS2:-0.95}"
EPS="${EPS:-1e-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EVAL_BATCHES="${EVAL_BATCHES:-10}"
LOG_EVERY="${LOG_EVERY:-50}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-2}"
PREFETCH_DEPTH="${PREFETCH_DEPTH:-8}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-float32}"
HEARTBEAT_EVERY_S="${HEARTBEAT_EVERY_S:-10}"

TOKEN_BUDGETS_CSV="${TOKEN_BUDGETS_CSV:-25000000,50000000,100000000,200000000}"
MODEL_SPECS_CSV="${MODEL_SPECS_CSV:-small:256:4:4:768,medium:384:6:6:1024,large:512:8:8:1536}"
IFS="," read -r -a TOKEN_BUDGETS <<< "$TOKEN_BUDGETS_CSV"
IFS="," read -r -a MODEL_SPECS <<< "$MODEL_SPECS_CSV"

if [[ "$WANDB_MODE" != "disabled" ]]; then
  if [[ -z "$WANDB_PROJECT" || -z "$WANDB_ENTITY" ]]; then
    echo "WANDB_PROJECT and WANDB_ENTITY must be set unless WANDB_MODE=disabled" >&2
    exit 1
  fi
fi

if [[ -z "$DATASET_ARTIFACT" ]]; then
  if [[ ! -f "$TRAIN_TOKENS_PATH" || ! -f "$VAL_TOKENS_PATH" ]]; then
    echo "Local train/val token files were not found. Set DATASET_ARTIFACT or point TRAIN_TOKENS_PATH / VAL_TOKENS_PATH to valid files." >&2
    exit 1
  fi
fi

mkdir -p "$SWEEP_DIR" "$BASE_SCRATCH"

cat > "$SWEEP_DIR/README.txt" <<EOF
Sweep name: $SWEEP_NAME
Project root: $PROJECT_ROOT
Created: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
W&B mode: $WANDB_MODE
W&B project: $WANDB_PROJECT
W&B entity: $WANDB_ENTITY
Dataset artifact: ${DATASET_ARTIFACT:-<local-files>}
Train tokens path: $TRAIN_TOKENS_PATH
Val tokens path: $VAL_TOKENS_PATH
Vocab json path: $VOCAB_JSON_PATH
Context length: $CONTEXT_LENGTH
Batch size: $BATCH_SIZE
Device: $DEVICE
Dtype: $DTYPE
EOF

printf "run_name\tsize\td_model\tnum_layers\tnum_heads\td_ff\ttarget_tokens\tmax_iters\twarmup_iters\teval_every\tsave_every\n" > "$SWEEP_DIR/sweep_plan.tsv"

cleanup_run_scratch() {
  local scratch_dir="$1"
  rm -rf "$scratch_dir"
  sync || true
  sleep 2
}

for spec in "${MODEL_SPECS[@]}"; do
  IFS=":" read -r SIZE_NAME D_MODEL NUM_LAYERS NUM_HEADS D_FF <<< "$spec"

  for TARGET_TOKENS in "${TOKEN_BUDGETS[@]}"; do
    TOKENS_PER_STEP=$((BATCH_SIZE * CONTEXT_LENGTH))
    MAX_ITERS=$(((TARGET_TOKENS + TOKENS_PER_STEP - 1) / TOKENS_PER_STEP))
    WARMUP_ITERS=$((MAX_ITERS / 50))
    if (( WARMUP_ITERS < 50 )); then
      WARMUP_ITERS=50
    fi
    if (( WARMUP_ITERS > MAX_ITERS )); then
      WARMUP_ITERS=$MAX_ITERS
    fi

    EVAL_EVERY=$((MAX_ITERS / 10))
    if (( EVAL_EVERY < 100 )); then
      EVAL_EVERY=100
    fi
    if (( EVAL_EVERY > MAX_ITERS )); then
      EVAL_EVERY=$MAX_ITERS
    fi
    SAVE_EVERY="$EVAL_EVERY"

    TOKENS_M=$((TARGET_TOKENS / 1000000))
    RUN_NAME="${SIZE_NAME}_tok${TOKENS_M}M_d${D_MODEL}_l${NUM_LAYERS}_h${NUM_HEADS}"
    RUN_DIR="$SWEEP_DIR/$RUN_NAME"
    SCRATCH_DIR="$BASE_SCRATCH/$RUN_NAME"

    mkdir -p "$RUN_DIR"
    cleanup_run_scratch "$SCRATCH_DIR"
    mkdir -p "$SCRATCH_DIR"

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$RUN_NAME" "$SIZE_NAME" "$D_MODEL" "$NUM_LAYERS" "$NUM_HEADS" "$D_FF" \
      "$TARGET_TOKENS" "$MAX_ITERS" "$WARMUP_ITERS" "$EVAL_EVERY" "$SAVE_EVERY" \
      >> "$SWEEP_DIR/sweep_plan.tsv"

    CMD=(
      python3 scripts/train_lm.py
      --context_length "$CONTEXT_LENGTH"
      --batch_size "$BATCH_SIZE"
      --d_model "$D_MODEL"
      --num_layers "$NUM_LAYERS"
      --num_heads "$NUM_HEADS"
      --d_ff "$D_FF"
      --max_lr "$MAX_LR"
      --min_lr "$MIN_LR"
      --betas1 "$BETAS1"
      --betas2 "$BETAS2"
      --eps "$EPS"
      --weight_decay "$WEIGHT_DECAY"
      --grad_clip "$GRAD_CLIP"
      --max_iters "$MAX_ITERS"
      --target_tokens_seen "$TARGET_TOKENS"
      --warmup_iters "$WARMUP_ITERS"
      --cosine_cycle_iters "$MAX_ITERS"
      --log_every "$LOG_EVERY"
      --eval_every "$EVAL_EVERY"
      --eval_batches "$EVAL_BATCHES"
      --save_every "$SAVE_EVERY"
      --prefetch_workers "$PREFETCH_WORKERS"
      --prefetch_depth "$PREFETCH_DEPTH"
      --heartbeat_every_s "$HEARTBEAT_EVERY_S"
      --device "$DEVICE"
      --dtype "$DTYPE"
      --scratch_dir "$SCRATCH_DIR"
      --wandb_mode "$WANDB_MODE"
      --wandb_group "$WANDB_GROUP"
      --wandb_run_name "$RUN_NAME"
      --wandb_tags "scaling,compute_optimal,size-${SIZE_NAME},tokens-${TOKENS_M}M"
      --checkpoint_artifact_name "${WANDB_PROJECT:-local}-${RUN_NAME}-checkpoints"
      --checkpoint_keep_milestone_every "$SAVE_EVERY"
      --run_record_dir "$RUN_DIR"
    )

    if [[ "$WANDB_MODE" != "disabled" ]]; then
      CMD+=(--wandb_project "$WANDB_PROJECT" --wandb_entity "$WANDB_ENTITY")
    fi

    if [[ -n "$DATASET_ARTIFACT" ]]; then
      CMD+=(--dataset_artifact "$DATASET_ARTIFACT")
    else
      CMD+=(
        --train_tokens_path "$TRAIN_TOKENS_PATH"
        --val_tokens_path "$VAL_TOKENS_PATH"
        --vocab_json_path "$VOCAB_JSON_PATH"
      )
    fi

    printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
    printf '\n' >> "$RUN_DIR/command.sh"
    chmod +x "$RUN_DIR/command.sh"

    echo "[run] $RUN_NAME"
    "${CMD[@]}" 2>&1 | tee "$RUN_DIR/stdout.log"

    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$RUN_DIR/completed_at_utc.txt"
    cleanup_run_scratch "$SCRATCH_DIR"
  done
done

echo "[done] sweep finished -> $SWEEP_DIR"
