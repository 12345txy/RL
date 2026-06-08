#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

# --- defaults (override via env or positional args) ---
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-0:5}"
WORKERS="${WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-results/swebench_cloud}"
MODEL="${MODEL:-hosted_vllm/grpo}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8000/v1}"
CONFIG="${CONFIG:-configs/swebench_modal_vllm.yaml}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_swebench_cloud.sh [options]

Cloud SWE-bench agent run (Modal sandbox + local/cloud model).

Options (env vars):
  SUBSET=verified          swe-bench subset (verified|lite|...)
  SPLIT=test               dataset split
  SLICE=0:5                instance slice (smoke test: first 5)
  WORKERS=4                parallel workers
  OUTPUT_DIR=...           output directory (contains preds.json)
  MODEL=...                litellm model name
  VLLM_BASE=...            vLLM OpenAI API base (for hosted_vllm/*)
  CONFIG=...               extra yaml merged after swebench.yaml

Examples:
  # Smoke test with local vLLM + Modal sandbox
  bash scripts/run_swebench_cloud.sh

  # Use OpenAI API model instead of local vLLM
  MODEL=openai/gpt-4o bash scripts/run_swebench_cloud.sh

  # Run 20 instances
  SLICE=0:20 WORKERS=8 bash scripts/run_swebench_cloud.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v mini-extra >/dev/null 2>&1; then
  echo "mini-extra not found. Run: bash scripts/setup_swebench_cloud.sh" >&2
  exit 1
fi

if ! modal profile list 2>/dev/null | grep -q .; then
  echo "Modal not authenticated. Run: modal setup" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Warn if using local vLLM but server may be down.
if [[ "$MODEL" == hosted_vllm/* ]]; then
  if ! curl -sf "${VLLM_BASE%/}/models" >/dev/null 2>&1; then
    echo "WARNING: vLLM not reachable at ${VLLM_BASE}" >&2
    echo "Start it first, e.g.:" >&2
    echo "  vllm serve models/Qwen3.5-2B --port 8000" >&2
    echo "  # or with LoRA:" >&2
    echo "  vllm serve models/Qwen3.5-2B --enable-lora --lora-modules grpo=outputs/grpo-mbpp-qwen35-2b-beta --port 8000" >&2
  fi
fi

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

echo "==> SWE-bench cloud run"
echo "    subset=$SUBSET split=$SPLIT slice=$SLICE workers=$WORKERS"
echo "    model=$MODEL output=$OUTPUT_DIR"

mini-extra swebench \
  -c swebench \
  -c "$CONFIG" \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --slice "$SLICE" \
  --workers "$WORKERS" \
  --model "$MODEL" \
  -o "$OUTPUT_DIR"

PREDS="$OUTPUT_DIR/preds.json"
if [[ -f "$PREDS" ]]; then
  echo
  echo "==> Done. Predictions: $PREDS"
  echo "    Evaluate: bash scripts/eval_swebench_sbcli.sh $PREDS <run_id>"
else
  echo "WARNING: preds.json not found under $OUTPUT_DIR" >&2
  exit 1
fi
