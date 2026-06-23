#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
SLICE="${SLICE:-10:30}"
WORKERS="${WORKERS:-4}"
REDO_EXISTING="${REDO_EXISTING:-1}"
MODEL="${MODEL:-hosted_vllm/Qwen3.6-27B}"
MODEL_SLUG="${MODEL##*/}"
OUTPUT_DIR="${OUTPUT_DIR:-results/swebench_vm_docker/${MODEL_SLUG}-${SLICE}}"
VLLM_BASE="${VLLM_BASE:-https://sv-1de238dd-3e5f-4057-96ff-e7820166f5d1-8000-x-defau-c9b4bcae2d.sproxy.bj-14.alayanew.com:22443/v1}"
CONFIG="${CONFIG:-configs/swebench_docker_gemma4_12b.yaml}"
CONDA_ENV="${CONDA_ENV:-swebench}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_swebench_vm_docker.sh

Run mini-swe-agent on SWE-bench using local Docker on this VM.
vLLM must be reachable at VLLM_BASE (local GPU box or SSH tunnel).

Env:
  SUBSET=verified          SWE-bench subset
  SPLIT=test
  SLICE=0:100              instance slice
  WORKERS=2                parallel instances (CPU/RAM bound)
  OUTPUT_DIR=results/swebench_vm_docker/<model>  (default, derived from MODEL)
  VLLM_BASE=https://sv-1de238dd-3e5f-4057-96ff-e7820166f5d1-8000-x-defau-c9b4bcae2d.sproxy.bj-14.alayanew.com:22443/v1
  MODEL=hosted_vllm/Qwen3.6-27B
  REDO_EXISTING=1          overwrite existing trajectories
  MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=2   LLM retries on timeout (2 attempts = 1 retry)
  CONDA_ENV=swebench

Example:
  VLLM_BASE=https://sv-1de238dd-3e5f-4057-96ff-e7820166f5d1-8000-x-defau-c9b4bcae2d.sproxy.bj-14.alayanew.com:22443/v1 SLICE=0:100 WORKERS=2 \
    bash scripts/run_swebench_vm_docker.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1 || ! docker ps >/dev/null 2>&1; then
  echo "ERROR: Docker is not usable. See docs/swebench_vm_docker_guide.md" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if ! command -v mini-extra >/dev/null 2>&1; then
  echo "mini-extra not found. Run: bash scripts/setup_swebench_vm.sh" >&2
  exit 1
fi

if ! curl -sf "${VLLM_BASE%/}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM not reachable at ${VLLM_BASE}" >&2
  echo "Start vLLM on the GPU machine and/or open an SSH tunnel:" >&2
  echo "  ssh -N -L 8000:127.0.0.1:8000 user@GPU_HOST" >&2
  exit 1
fi

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="${MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT:-2}"

mkdir -p "$OUTPUT_DIR"

echo "==> SWE-bench (VM Docker + vLLM)"
echo "    subset=$SUBSET split=$SPLIT slice=$SLICE workers=$WORKERS"
echo "    model=$MODEL vllm=$VLLM_BASE output=$OUTPUT_DIR"
echo "    llm_retry_attempts=$MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"

REDO_ARG=()
if [[ "$REDO_EXISTING" == "1" ]]; then
  REDO_ARG=(--redo-existing)
fi

mini-extra swebench \
  -c swebench \
  -c "$CONFIG" \
  -c "model.model_kwargs.api_base=${VLLM_BASE}" \
  --environment-class docker \
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
  echo "    Local eval: bash scripts/eval_swebench_local.sh $PREDS vm-docker-run"
else
  echo "WARNING: preds.json not found" >&2
  exit 1
fi
