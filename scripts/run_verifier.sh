#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
RL_DIR="${RL_DIR:-outputs/rl-gemma4-12b-miniswe-rl2}"
VERIFIER_OUT="${VERIFIER_OUT:-outputs/verifier-gemma4-12b}"
BON_K="${BON_K:-8}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8000/v1}"
MODEL="${MODEL:-gemma-4-12B-it}"
OUTPUT_DIR="${OUTPUT_DIR:-results/verified_final}"
DEV_SPLIT="${DEV_SPLIT:-}"

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_verifier.sh

Phase 4: train verifier + Best-of-N Verified evaluation.

Env:
  RL_DIR=outputs/rl-gemma4-12b-miniswe-rl2
  VERIFIER_OUT=outputs/verifier-gemma4-12b
  BON_K=8
  OUTPUT_DIR=results/verified_final
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

POS_TRAJ="$RL_DIR/rl_positive_trajectories.jsonl"
ROLL_DIR="$RL_DIR/rollouts"
INPUTS=()
if [[ -f "$POS_TRAJ" ]]; then
  INPUTS+=("$POS_TRAJ")
fi
if [[ -d "$ROLL_DIR" ]]; then
  while IFS= read -r -d '' f; do
    INPUTS+=("$f")
  done < <(find "$ROLL_DIR" -name 'sft_buffer_step*.jsonl' -print0 2>/dev/null || true)
fi

if [[ ${#INPUTS[@]} -eq 0 ]]; then
  echo "WARN: no RL trajectories found; using merged SFT data for verifier demo"
  INPUTS=("data/sft/sft_merged.jsonl")
fi

echo "==> Training verifier from ${#INPUTS[@]} source(s)"
python scripts/train_verifier.py \
  --input "${INPUTS[@]}" \
  --output_dir "$VERIFIER_OUT"

echo "==> Best-of-$BON_K Verified evaluation"
mkdir -p "$OUTPUT_DIR"
python scripts/best_of_n_eval.py \
  --output_dir "$OUTPUT_DIR" \
  --vllm_base "$VLLM_BASE" \
  --model "$MODEL" \
  --verifier "$VERIFIER_OUT" \
  --k "$BON_K" \
  ${DEV_SPLIT:+--dev_split "$DEV_SPLIT"}

PREDS="$OUTPUT_DIR/preds.json"
if [[ -f "$PREDS" ]]; then
  RUN_ID="verified-final-$(date +%Y%m%d-%H%M%S)"
  echo "==> Harness eval"
  bash scripts/eval_swebench_local.sh "$PREDS" "$RUN_ID" || true
fi
