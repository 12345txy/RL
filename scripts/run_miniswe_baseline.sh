#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
MODEL_PATH="${MODEL_PATH:-models/gemma-4-12B-it}"
VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8000/v1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-12B-it}"
SLICE="${SLICE:-0:10}"
OUTPUT_DIR="${OUTPUT_DIR:-results/miniswe_baseline}"
SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-0}"
SKIP_TOOLCHECK="${SKIP_TOOLCHECK:-0}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_miniswe_baseline.sh

Phase 0: verify Gemma4 tool calls + run small mini-swe-agent SWE-bench slice baseline.

Env:
  MODEL_PATH=models/gemma-4-12B-it
  VLLM_BASE=http://127.0.0.1:8000/v1
  SLICE=0:10                 Quick baseline slice (use 0:500 for full Verified)
  OUTPUT_DIR=results/miniswe_baseline
  SKIP_VLLM_CHECK=1          Skip curl probe
  SKIP_TOOLCHECK=1           Skip python tool-call probe

Prerequisites:
  bash scripts/setup_swebench_vm.sh
  bash scripts/serve_gemma4_12b.sh   (separate terminal)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

mkdir -p data/splits "$OUTPUT_DIR"

if [[ ! -f data/splits/verified_dev_100.json ]]; then
  echo "==> Creating Verified dev split (data/splits/verified_dev_100.json)"
  python - <<'PY'
from data.swe_utils import make_verified_dev_split
make_verified_dev_split(n=100, output_path="data/splits/verified_dev_100.json")
print("Wrote data/splits/verified_dev_100.json")
PY
fi

if [[ "$SKIP_VLLM_CHECK" != "1" ]]; then
  if ! curl -sf "${VLLM_BASE%/}/models" >/dev/null; then
    echo "ERROR: vLLM not reachable at $VLLM_BASE" >&2
    echo "Start: bash scripts/serve_gemma4_12b.sh" >&2
    exit 1
  fi
fi

if [[ "$SKIP_TOOLCHECK" != "1" ]]; then
  echo "==> Gemma4 tool-call compatibility check"
  python scripts/verify_gemma4_toolcall.py \
    --api_base "$VLLM_BASE" \
    --model "$SERVED_MODEL_NAME"
fi

echo "==> Running mini-swe-agent SWE-bench eval slice=$SLICE"
VLLM_BASE="$VLLM_BASE" SLICE="$SLICE" OUTPUT_DIR="$OUTPUT_DIR" \
  bash scripts/run_swebench_vm_docker.sh

echo
echo "==> Baseline artifacts under $OUTPUT_DIR"
echo "    Next: bash scripts/run_prepare_data.sh"
