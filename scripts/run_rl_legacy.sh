#!/usr/bin/env bash
# Legacy pseudo-RL (rollout + SFT refresh). Prefer: bash scripts/run_rl_skyrl.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
CONDA_ENV="${CONDA_ENV:-RL}"
STAGE="${STAGE:-rl1}"
MODEL="${MODEL:-gemma-4-12B-it}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8000/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl-gemma4-12b-miniswe}"
SKIP_SFT_CONTINUE="${SKIP_SFT_CONTINUE:-0}"
source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
RL_POOL="data/rl/swegym_rl_train.jsonl"
STEPS=500
if [[ "$STAGE" == "rl2" ]]; then STEPS=2000; fi
OUT="${OUTPUT_DIR}-${STAGE}"
python scripts/train_agent_rl.py \
  --rl_pool "$RL_POOL" --output_dir "$OUT" --api_base "$VLLM_BASE" \
  --model "$MODEL" --steps "$STEPS" --batch_size 8 --num_rollouts 4 --max_turns 50 --save_every 50
if [[ "$SKIP_SFT_CONTINUE" != "1" && -s "${OUT}/rl_positive_trajectories.jsonl" ]]; then
  TRAIN_PATH="${OUT}/rl_positive_trajectories.jsonl" STAGE=lora OUTPUT_DIR="${OUT}-refresh" bash scripts/run_sft.sh
fi
