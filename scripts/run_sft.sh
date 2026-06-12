#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export USE_FLA="${USE_FLA:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-swe-rl}"
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
NO_SWANLAB="${NO_SWANLAB:-0}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-}"
SWANLAB_DESCRIPTION="${SWANLAB_DESCRIPTION:-Gemma4-12B mini-swe-agent SFT}"
LOSS_TYPE="${LOSS_TYPE:-chunked_nll}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-50}"
DDP_TIMEOUT="${DDP_TIMEOUT:-1200}"

MODEL_PATH="${MODEL_PATH:-models/gemma-4-12B-it}"
TRAIN_PATH="${TRAIN_PATH:-data/sft/sft_merged.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft-gemma4-12b-miniswe}"
STAGE="${STAGE:-lora}"
SKIP_DATA="${SKIP_DATA:-0}"
SKIP_SEQ_ANALYSIS="${SKIP_SEQ_ANALYSIS:-0}"
MAX_SEQ_CAP="${MAX_SEQ_CAP:-28672}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-}"
SEQ_STATS_PATH="${SEQ_STATS_PATH:-data/sft/seq_length_stats.json}"
DEEPSPEED_ZERO_STAGE="${DEEPSPEED_ZERO_STAGE:-3}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
ACCEL_CONFIG="${ACCEL_CONFIG:-}"
if [[ -z "$ACCEL_CONFIG" ]]; then
  case "$DEEPSPEED_ZERO_STAGE" in
    1) ACCEL_CONFIG=configs/accelerate_deepspeed_zero1_8gpu.yaml ;;
    3) ACCEL_CONFIG=configs/accelerate_deepspeed_zero3_8gpu.yaml ;;
    *) ACCEL_CONFIG=configs/accelerate_deepspeed_zero2_8gpu.yaml ;;
  esac
fi
PREPROCESS_CACHE_DIR="${PREPROCESS_CACHE_DIR:-data/sft/preprocessed}"
USE_PREPROCESSED="${USE_PREPROCESSED:-1}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
PREPROCESS_NUM_PROC="${PREPROCESS_NUM_PROC:-8}"

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate RL

usage() {
  cat <<'EOF'
Usage: bash scripts/run_sft.sh

Phase 2: two-stage SFT (LoRA debug -> full) on mini-swe-agent trajectories.

Env:
  STAGE=lora|full
  MODEL_PATH=models/gemma-4-12B-it
  TRAIN_PATH=data/sft/sft_merged.jsonl
  OUTPUT_DIR=outputs/sft-gemma4-12b-miniswe
  SKIP_DATA=1
  SKIP_SEQ_ANALYSIS=1
  MAX_SEQ_LENGTH=28672
  MAX_SEQ_CAP=28672
  PREPROCESS_CACHE_DIR=data/sft/preprocessed
  USE_PREPROCESSED=1
  FORCE_PREPROCESS=0
  PREPROCESS_NUM_PROC=8
  DEEPSPEED_ZERO_STAGE=1|2|3   Default: 3 (ZeRO-3); auto-picks accelerate config when ACCEL_CONFIG unset
  GRADIENT_CHECKPOINTING=1       Default: 1 (on)
  ACCEL_CONFIG=                 Override accelerate yaml (optional)
  SWANLAB_PROJECT=swe-rl
  SWANLAB_MODE=local
  NO_SWANLAB=1
  LOGGING_STEPS=1
  SAVE_STEPS=50
  DDP_TIMEOUT=1200
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$SKIP_DATA" != "1" && ! -f "$TRAIN_PATH" ]]; then
  echo "==> Missing $TRAIN_PATH; running data prep"
  bash scripts/run_prepare_data.sh
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  echo "Run: bash scripts/download_gemma4_12b.sh" >&2
  exit 1
fi

EXTRA_ARGS=()
OUT="$OUTPUT_DIR"
case "$STAGE" in
  lora)
    OUT="${OUTPUT_DIR}-lora"
    EXTRA_ARGS+=(--lora_r 64 --lora_alpha 128 --learning_rate 2e-5 --num_train_epochs 2)
    ;;
  full)
    OUT="${OUTPUT_DIR}-full"
    EXTRA_ARGS+=(--full_finetune --learning_rate 1e-5 --num_train_epochs 1)
    ;;
  *)
    echo "Unknown STAGE=$STAGE (use lora or full)" >&2
    exit 1
    ;;
esac

if [[ -z "$SWANLAB_EXPERIMENT_NAME" ]]; then
  SWANLAB_EXPERIMENT_NAME="sft-gemma4-12b-${STAGE}"
fi

SWANLAB_ARGS=(
  --swanlab_project "$SWANLAB_PROJECT"
  --swanlab_experiment_name "$SWANLAB_EXPERIMENT_NAME"
  --swanlab_description "$SWANLAB_DESCRIPTION"
)
if [[ "$NO_SWANLAB" == "1" ]]; then
  SWANLAB_ARGS+=(--no_swanlab)
fi

if [[ -z "$MAX_SEQ_LENGTH" ]]; then
  if [[ "$SKIP_SEQ_ANALYSIS" != "1" ]]; then
    echo "==> Analyzing SFT token lengths (MAX_SEQ_CAP=$MAX_SEQ_CAP)"
    python scripts/analyze_sft_seq_lengths.py \
      --model_path "$MODEL_PATH" \
      --train_path "$TRAIN_PATH" \
      --output_path "$SEQ_STATS_PATH" \
      --max_cap "$MAX_SEQ_CAP"
  fi
  if [[ ! -f "$SEQ_STATS_PATH" ]]; then
    echo "ERROR: missing $SEQ_STATS_PATH; run analyze_sft_seq_lengths.py or set MAX_SEQ_LENGTH" >&2
    exit 1
  fi
  MAX_SEQ_LENGTH="$(python - <<PY
import json
print(json.load(open("${SEQ_STATS_PATH}", encoding="utf-8"))["recommended_max_seq_length"])
PY
)"
fi

echo "==> SFT stage=$STAGE output=$OUT max_seq_length=$MAX_SEQ_LENGTH deepspeed_zero=$DEEPSPEED_ZERO_STAGE grad_ckpt=$GRADIENT_CHECKPOINTING save_steps=$SAVE_STEPS logging_steps=$LOGGING_STEPS ddp_timeout=${DDP_TIMEOUT}s preprocess_cache=$PREPROCESS_CACHE_DIR use_preprocessed=$USE_PREPROCESSED swanlab=$([[ "$NO_SWANLAB" == "1" ]] && echo off || echo on) project=$SWANLAB_PROJECT run=$SWANLAB_EXPERIMENT_NAME"
PREPROCESS_ARGS=()
if [[ "$USE_PREPROCESSED" == "1" ]]; then
  PREPROCESS_ARGS+=(--use_preprocessed)
else
  PREPROCESS_ARGS+=(--no-use_preprocessed)
fi
if [[ "$FORCE_PREPROCESS" == "1" ]]; then
  PREPROCESS_ARGS+=(--force_preprocess)
fi
GRAD_CKPT_ARGS=(--gradient_checkpointing)
if [[ "$GRADIENT_CHECKPOINTING" != "1" ]]; then
  GRAD_CKPT_ARGS=(--no-gradient_checkpointing)
fi

accelerate launch --config_file "$ACCEL_CONFIG" scripts/train_sft.py \
  --model_path "$MODEL_PATH" \
  --train_path "$TRAIN_PATH" \
  --output_dir "$OUT" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --preprocess_cache_dir "$PREPROCESS_CACHE_DIR" \
  --preprocess_num_proc "$PREPROCESS_NUM_PROC" \
  "${PREPROCESS_ARGS[@]}" \
  "${GRAD_CKPT_ARGS[@]}" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --loss_type "$LOSS_TYPE" \
  --logging_steps "$LOGGING_STEPS" \
  --save_steps "$SAVE_STEPS" \
  --deepspeed_zero_stage "$DEEPSPEED_ZERO_STAGE" \
  --ddp_timeout "$DDP_TIMEOUT" \
  "${SWANLAB_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
