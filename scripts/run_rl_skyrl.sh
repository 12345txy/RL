#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
STAGE="${STAGE:-rl1}"
MODEL="${MODEL:-gemma-4-12B-it}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8000/v1}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-outputs/sft-gemma4-12b-miniswe-full}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl-gemma4-12b-miniswe}"
SKIP_SFT_CONTINUE="${SKIP_SFT_CONTINUE:-0}"

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_rl_skyrl.sh

Phase 3: Multi-turn Agent RL (rollout collection + policy refresh).

Stages:
  rl1  SWE-Gym Lite / small pool, 500 steps (default)
  rl2  Full RL pool, 2000 steps

Env:
  STAGE=rl1|rl2
  VLLM_BASE=http://127.0.0.1:8000/v1
  SFT_CHECKPOINT=outputs/sft-gemma4-12b-miniswe-full
  OUTPUT_DIR=outputs/rl-gemma4-12b-miniswe
  SKIP_SFT_CONTINUE=1

Prerequisites:
  vLLM serving SFT checkpoint (see serve_gemma4_12b.sh LORA_PATH/CHECKPOINT)
  data/rl/swegym_rl_train.jsonl from run_prepare_data.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! curl -sf "${VLLM_BASE%/}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM not reachable at $VLLM_BASE" >&2
  echo "Start: CHECKPOINT=$SFT_CHECKPOINT bash scripts/serve_gemma4_12b.sh" >&2
  exit 1
fi

RL_POOL="data/rl/swegym_rl_train.jsonl"
STEPS=500
if [[ "$STAGE" == "rl2" ]]; then
  STEPS=2000
elif [[ "$STAGE" == "rl1" ]]; then
  python data/prepare_swe_gym_sft.py --lite_only --rl_output data/rl/swegym_rl_train_lite.jsonl || true
  if [[ -f data/rl/swegym_rl_train_lite.jsonl ]]; then
    RL_POOL="data/rl/swegym_rl_train_lite.jsonl"
  fi
fi

OUT="${OUTPUT_DIR}-${STAGE}"
echo "==> Agent RL stage=$STAGE pool=$RL_POOL steps=$STEPS"
python scripts/train_agent_rl.py \
  --rl_pool "$RL_POOL" \
  --output_dir "$OUT" \
  --api_base "$VLLM_BASE" \
  --model "$MODEL" \
  --steps "$STEPS" \
  --batch_size 8 \
  --num_rollouts 4 \
  --max_turns 50 \
  --save_every 50

if [[ "$SKIP_SFT_CONTINUE" == "1" ]]; then
  exit 0
fi

POS="${OUT}/rl_positive_trajectories.jsonl"
if [[ -f "$POS" && -s "$POS" ]]; then
  echo "==> Policy refresh SFT on RL-positive trajectories"
  TRAIN_PATH="$POS" STAGE=lora OUTPUT_DIR="${OUT}-refresh" bash scripts/run_sft.sh
fi
