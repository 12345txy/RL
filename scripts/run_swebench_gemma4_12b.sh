#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-0:5}"
WORKERS="${WORKERS:-4}"
REDO_EXISTING="${REDO_EXISTING:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-results/swebench_gemma4_12b}"
MODEL="${MODEL:-hosted_vllm/gemma-4-12B-it}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8000/v1}"
CONFIG="${CONFIG:-configs/swebench_modal_gemma4_12b.yaml}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

if ! command -v mini-extra >/dev/null 2>&1; then
  echo "mini-extra not found. Run: bash scripts/setup_swebench_cloud.sh" >&2
  exit 1
fi

if ! modal profile list 2>/dev/null | grep -q .; then
  echo "Modal not authenticated. Run: modal setup" >&2
  exit 1
fi

if ! curl -sf "${VLLM_BASE%/}/models" >/dev/null 2>&1; then
  echo "vLLM not reachable at ${VLLM_BASE}" >&2
  echo "Start in another terminal: bash scripts/serve_gemma4_12b.sh" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "==> SWE-bench (Gemma4-12B + Modal)"
echo "    subset=$SUBSET split=$SPLIT slice=$SLICE workers=$WORKERS redo=$REDO_EXISTING"
echo "    model=$MODEL output=$OUTPUT_DIR"

REDO_ARG=()
if [[ "$REDO_EXISTING" == "1" ]]; then
  REDO_ARG=(--redo-existing)
fi

mini-extra swebench \
  -c swebench \
  -c "$CONFIG" \
  --environment-class swerex_modal \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --slice "$SLICE" \
  --workers "$WORKERS" \
  --model "$MODEL" \
  -o "$OUTPUT_DIR" \
  "${REDO_ARG[@]}"

PREDS="$OUTPUT_DIR/preds.json"
if [[ -f "$PREDS" ]]; then
  echo
  echo "==> Predictions: $PREDS"
  echo "    Evaluate: bash scripts/eval_swebench_sbcli.sh $PREDS gemma4-12b-verified"
else
  echo "WARNING: preds.json not found" >&2
  exit 1
fi
