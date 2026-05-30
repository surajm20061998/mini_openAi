#!/usr/bin/env bash
#
# Architecture-comparison sweep: Transformer vs Mamba (SSM) vs Linear-Attention
# (sub-quadratic), at matched active parameters and matched tokens seen.
#
# Structure mirrors `run_compute_optimal_scaling_sweep.sh`, with a third axis
# (architecture). For each (arch, size) we hand-tuned configs so that
# `model.param_count.count_active_params` is within ~10 % across arches at the
# same `size` — verify any time with:
#
#   python3 -c "from model.param_count import count_active_params as C; \
#       print(C('ssm', vocab_size=512, d_model=384, num_layers=6, d_state=24, \
#               d_conv=4, expand=4))"
#
# See EXPERIMENTS.md §3 (matched-compute methodology) and §4 (how it fits in).

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SWEEP_NAME="${SWEEP_NAME:-arch_comparison_$(date +%Y%m%d_%H%M%S)}"
SWEEP_DIR="${SWEEP_DIR:-$PROJECT_ROOT/sweep_experiments/$SWEEP_NAME}"
BASE_SCRATCH="${BASE_SCRATCH:-/tmp/mini_openai_sweeps/$SWEEP_NAME}"

WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-arch_comparison}"

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

# Capability evaluation (post-training) — set to 0 to skip.
RUN_CAPABILITY_EVAL="${RUN_CAPABILITY_EVAL:-1}"

# Sweep axes.
ARCHS_CSV="${ARCHS_CSV:-transformer,ssm,subquadratic}"
SIZES_CSV="${SIZES_CSV:-small,medium,large}"
TOKEN_BUDGETS_CSV="${TOKEN_BUDGETS_CSV:-25000000,50000000,100000000,200000000}"

IFS="," read -r -a ARCHS <<< "$ARCHS_CSV"
IFS="," read -r -a SIZES <<< "$SIZES_CSV"
IFS="," read -r -a TOKEN_BUDGETS <<< "$TOKEN_BUDGETS_CSV"

# ---------------------------------------------------------------------------
# Matched-active-params specs, hand-tuned via model/param_count.py.
# Each function echoes a space-separated config string consumed below.
# ---------------------------------------------------------------------------

emit_transformer_cfg() {
  # args: size
  case "$1" in
    small)  echo "d_model=256 num_layers=4 num_heads=4 d_ff=768"  ;;
    medium) echo "d_model=384 num_layers=6 num_heads=6 d_ff=1024" ;;
    large)  echo "d_model=512 num_layers=8 num_heads=8 d_ff=1536" ;;
    *) echo "unknown size: $1" >&2 ; return 1 ;;
  esac
}

emit_ssm_cfg() {
  # args: size — canonical Mamba d_state=16, expand=4. These match the
  # Transformer active-param totals within ~9 % at every size (verified with
  # model/param_count.py: small 2.5 %, medium 8.6 %, large 0.6 %).
  case "$1" in
    small)  echo "d_model=256 num_layers=4 num_heads=4 d_ff=768  ssm_d_state=16 ssm_d_conv=4 ssm_expand=4" ;;
    medium) echo "d_model=384 num_layers=6 num_heads=6 d_ff=1024 ssm_d_state=16 ssm_d_conv=4 ssm_expand=4" ;;
    large)  echo "d_model=512 num_layers=8 num_heads=8 d_ff=1536 ssm_d_state=16 ssm_d_conv=4 ssm_expand=4" ;;
    *) echo "unknown size: $1" >&2 ; return 1 ;;
  esac
}

emit_subquadratic_cfg() {
  # Linear attention shares the Transformer's parameter structure exactly.
  case "$1" in
    small)  echo "d_model=256 num_layers=4 num_heads=4 d_ff=768  subq_kind=linear subq_feature_map=elu" ;;
    medium) echo "d_model=384 num_layers=6 num_heads=6 d_ff=1024 subq_kind=linear subq_feature_map=elu" ;;
    large)  echo "d_model=512 num_layers=8 num_heads=8 d_ff=1536 subq_kind=linear subq_feature_map=elu" ;;
    *) echo "unknown size: $1" >&2 ; return 1 ;;
  esac
}

emit_cfg_for() {
  # args: arch size
  case "$1" in
    transformer)  emit_transformer_cfg  "$2" ;;
    ssm)          emit_ssm_cfg          "$2" ;;
    subquadratic) emit_subquadratic_cfg "$2" ;;
    *) echo "unknown arch: $1" >&2 ; return 1 ;;
  esac
}

# ---------------------------------------------------------------------------
# Sanity / preflight checks
# ---------------------------------------------------------------------------

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
Architectures: $ARCHS_CSV
Sizes: $SIZES_CSV
Token budgets: $TOKEN_BUDGETS_CSV
Run capability eval: $RUN_CAPABILITY_EVAL
EOF

printf "run_name\tarch\tsize\ttarget_tokens\tmax_iters\twarmup_iters\teval_every\tsave_every\tconfig\n" \
  > "$SWEEP_DIR/sweep_plan.tsv"

cleanup_run_scratch() {
  local scratch_dir="$1"
  rm -rf "$scratch_dir"
  sync || true
  sleep 2
}

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

for ARCH in "${ARCHS[@]}"; do
  for SIZE in "${SIZES[@]}"; do
    CFG_STR=$(emit_cfg_for "$ARCH" "$SIZE")

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
      RUN_NAME="${ARCH}_${SIZE}_tok${TOKENS_M}M"
      RUN_DIR="$SWEEP_DIR/$RUN_NAME"
      SCRATCH_DIR="$BASE_SCRATCH/$RUN_NAME"

      mkdir -p "$RUN_DIR"
      cleanup_run_scratch "$SCRATCH_DIR"
      mkdir -p "$SCRATCH_DIR"

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$RUN_NAME" "$ARCH" "$SIZE" "$TARGET_TOKENS" \
        "$MAX_ITERS" "$WARMUP_ITERS" "$EVAL_EVERY" "$SAVE_EVERY" \
        "$CFG_STR" >> "$SWEEP_DIR/sweep_plan.tsv"

      # Shared training command.
      CMD=(
        python3 scripts/train_lm.py
        --model_arch "$ARCH"
        --context_length "$CONTEXT_LENGTH"
        --batch_size "$BATCH_SIZE"
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
        --wandb_tags "arch_comparison,arch-${ARCH},size-${SIZE},tokens-${TOKENS_M}M"
        --checkpoint_artifact_name "${WANDB_PROJECT:-local}-${RUN_NAME}-checkpoints"
        --checkpoint_keep_milestone_every "$SAVE_EVERY"
        --run_record_dir "$RUN_DIR"
      )

      # Append the arch-specific `--key value` pairs from emit_cfg_for.
      for KV in $CFG_STR; do
        CMD+=(--"${KV%%=*}" "${KV#*=}")
      done

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

      # Optional capability evaluation on the just-trained checkpoint.
      if [[ "$RUN_CAPABILITY_EVAL" == "1" ]]; then
        CKPT_BEST="$SCRATCH_DIR/checkpoints/train_lm.best.pt"
        CKPT_LAST="$SCRATCH_DIR/checkpoints/train_lm.pt"
        EVAL_CKPT=""
        if [[ -f "$CKPT_BEST" ]]; then
          EVAL_CKPT="$CKPT_BEST"
        elif [[ -f "$CKPT_LAST" ]]; then
          EVAL_CKPT="$CKPT_LAST"
        fi
        if [[ -n "$EVAL_CKPT" ]]; then
          echo "[eval] $RUN_NAME -> capability suite on $EVAL_CKPT"
          python3 eval/run_capability_suite.py \
            --checkpoint "$EVAL_CKPT" \
            --resolved_config "$RUN_DIR/resolved_config.json" \
            --val_tokens_path "$VAL_TOKENS_PATH" \
            --vocab_size_from_resolved \
            --output_json "$RUN_DIR/capability.json" \
            --device "$DEVICE" \
            --dtype "$DTYPE" 2>&1 | tee -a "$RUN_DIR/stdout.log" || \
              echo "[eval] capability suite failed (non-fatal)"
        else
          echo "[eval] no checkpoint found in $SCRATCH_DIR/checkpoints — skipping capability suite"
        fi
      fi

      cleanup_run_scratch "$SCRATCH_DIR"
    done
  done
done

echo "[done] sweep finished -> $SWEEP_DIR"
