#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-models/gemma-4-12B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
PORT="${PORT:-8000}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.90}"
# SWE-bench is text-only; skip image/audio profiling for 12B Unified.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 0, \"audio\": 0}}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

export VLLM_USE_DEEP_GEMM=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  echo "Run: bash scripts/download_gemma4_12b.sh" >&2
  exit 1
fi

if [[ "$(basename "$MODEL_PATH")" == "gemma-4-12B-it" ]]; then
  if ! python - <<'PY'
import os, vllm
ok = os.path.exists(os.path.join(os.path.dirname(vllm.__file__), "model_executor/models/gemma4_unified.py"))
raise SystemExit(0 if ok else 1)
PY
  then
    echo "ERROR: gemma-4-12B-it (Unified) needs vLLM nightly (PR #44429)." >&2
    echo "  Stable vLLM 0.22.x hits: [4096] X [8192, 3840] linear mismatch." >&2
    echo "  Install nightly, e.g.:" >&2
    echo "    pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130" >&2
    echo "  Or Docker: vllm/vllm-openai:gemma4-unified" >&2
    exit 1
  fi
fi

echo "==> vLLM serve: $MODEL_PATH (served_name=$SERVED_MODEL_NAME port=$PORT tp=$TP max_len=$MAX_MODEL_LEN)"
echo "    tool_call_parser=gemma4 (required by mini-swe-agent)"
echo "    limit_mm_per_prompt=$LIMIT_MM_PER_PROMPT"

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --port "$PORT" \
  --dtype bfloat16 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM"
