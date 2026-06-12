#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:-models/gemma-4-26B-A4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}"
PORT="${PORT:-8000}"
BASE_PORT="${BASE_PORT:-8000}"
NUM_INSTANCES="${NUM_INSTANCES:-1}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.90}"
# SWE-bench is text-only; skip image/audio profiling.
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 0, \"audio\": 0}}"
LOG_DIR="${LOG_DIR:-results/vllm_gemma4_26b_a4b}"
PID_DIR="${PID_DIR:-results/vllm_gemma4_26b_a4b/pids}"
STOP="${STOP:-0}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

export VLLM_USE_DEEP_GEMM=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

usage() {
  cat <<'EOF'
Usage: bash scripts/serve_gemma4_26b_a4b.sh

Start vLLM for Gemma4-26B-A4B-it (MoE, OpenAI-compatible API, gemma4 tool parser).

Single instance (default):
  bash scripts/serve_gemma4_26b_a4b.sh

Multi-GPU tensor parallel (recommended if OOM on one card):
  TENSOR_PARALLEL_SIZE=2 bash scripts/serve_gemma4_26b_a4b.sh

Stop multi-instance background servers:
  STOP=1 NUM_INSTANCES=8 bash scripts/serve_gemma4_26b_a4b.sh

Env:
  MODEL_PATH=models/gemma-4-26B-A4B-it
  NUM_INSTANCES=1          MoE ~49GB: keep at 1; do not run one copy per GPU
  TENSOR_PARALLEL_SIZE=1   Use 2+ if single-GPU OOM
  MAX_MODEL_LEN=131072       (model supports up to 262144)
  GPU_MEMORY_UTILIZATION=0.90
  PORT=8000

Note: 26B-A4B is Gemma4ForConditionalGeneration (MoE), not 12B Unified.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

stop_instances() {
  local n="${1:-$NUM_INSTANCES}"
  local stopped=0
  mkdir -p "$PID_DIR"
  for ((i = 0; i < n; i++)); do
    local pid_file="$PID_DIR/gpu${i}.pid"
    if [[ -f "$pid_file" ]]; then
      local pid
      pid="$(cat "$pid_file")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "==> Stopping GPU $i (pid $pid, port $((BASE_PORT + i)))"
        kill "$pid" 2>/dev/null || true
        stopped=$((stopped + 1))
      fi
      rm -f "$pid_file"
    fi
  done
  if [[ "$stopped" -eq 0 ]]; then
    echo "No running instances found under $PID_DIR"
  fi
}

if [[ "$STOP" == "1" ]]; then
  stop_instances "$NUM_INSTANCES"
  exit 0
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  echo "Run: bash scripts/download_gemma4_26b_a4b.sh" >&2
  exit 1
fi

if [[ "$(basename "$MODEL_PATH")" == "gemma-4-26B-A4B-it" ]]; then
  if ! python - <<'PY'
import os, vllm
root = os.path.join(os.path.dirname(vllm.__file__), "model_executor/models")
ok = os.path.exists(os.path.join(root, "gemma4_mm.py")) and os.path.exists(
    os.path.join(root, "gemma4.py")
)
raise SystemExit(0 if ok else 1)
PY
  then
    echo "ERROR: gemma-4-26B-A4B-it (MoE) needs vLLM with Gemma4 MM/MoE support." >&2
    echo "  Install nightly, e.g.:" >&2
    echo "    pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly/cu130" >&2
    exit 1
  fi
fi

if [[ "$NUM_INSTANCES" -gt 1 ]]; then
  echo "WARNING: 26B-A4B MoE (~49GB weights) rarely fits one copy per GPU." >&2
  echo "         Prefer NUM_INSTANCES=1 with TENSOR_PARALLEL_SIZE=2+ instead." >&2
fi

if [[ "$NUM_INSTANCES" -gt 1 && "$TP" -ne 1 ]]; then
  echo "ERROR: multi-instance mode requires TENSOR_PARALLEL_SIZE=1 (one model copy per GPU)." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if [[ "$NUM_INSTANCES" -gt 1 && "$NUM_INSTANCES" -gt "$GPU_COUNT" ]]; then
    echo "ERROR: NUM_INSTANCES=$NUM_INSTANCES but only $GPU_COUNT GPU(s) visible." >&2
    exit 1
  fi
  if [[ "$NUM_INSTANCES" -le 1 && "$TP" -gt "$GPU_COUNT" ]]; then
    echo "ERROR: TENSOR_PARALLEL_SIZE=$TP but only $GPU_COUNT GPU(s) visible." >&2
    exit 1
  fi
fi

run_vllm() {
  local gpu_id="$1"
  local listen_port="$2"
  local log_file="$3"

  echo "==> Starting vLLM on GPU $gpu_id port $listen_port (log: $log_file)"
  CUDA_VISIBLE_DEVICES="$gpu_id" vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$listen_port" \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser gemma4 \
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" \
    --tensor-parallel-size 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM" \
    >>"$log_file" 2>&1 &
  echo "$!" > "$PID_DIR/gpu${gpu_id}.pid"
}

if [[ "$NUM_INSTANCES" -le 1 ]]; then
  echo "==> vLLM serve: $MODEL_PATH (served_name=$SERVED_MODEL_NAME port=$PORT tp=$TP max_len=$MAX_MODEL_LEN)"
  echo "    architecture=Gemma4ForConditionalGeneration (26B-A4B MoE)"
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
  exit 0
fi

mkdir -p "$LOG_DIR" "$PID_DIR"
stop_instances "$NUM_INSTANCES" || true
sleep 1

echo "==> vLLM x$NUM_INSTANCES: $MODEL_PATH"
echo "    served_name=$SERVED_MODEL_NAME base_port=$BASE_PORT max_len=$MAX_MODEL_LEN gpu_mem=$GPU_MEM"
echo "    one independent instance per GPU (TP=1 each)"
echo "    logs: $LOG_DIR  pids: $PID_DIR"

for ((i = 0; i < NUM_INSTANCES; i++)); do
  run_vllm "$i" "$((BASE_PORT + i))" "$LOG_DIR/gpu${i}.log"
done

echo
echo "==> Waiting for APIs (may take several minutes while models load)..."
ready=0
for _ in $(seq 1 120); do
  ready=0
  for ((i = 0; i < NUM_INSTANCES; i++)); do
    if curl -sf "http://127.0.0.1:$((BASE_PORT + i))/v1/models" >/dev/null 2>&1; then
      ready=$((ready + 1))
    fi
  done
  if [[ "$ready" -eq "$NUM_INSTANCES" ]]; then
    break
  fi
  sleep 5
done

echo "==> Ready: $ready / $NUM_INSTANCES instances"
for ((i = 0; i < NUM_INSTANCES; i++)); do
  echo "    GPU $i: http://127.0.0.1:$((BASE_PORT + i))/v1"
done
echo
echo "Stop all: STOP=1 NUM_INSTANCES=$NUM_INSTANCES bash scripts/serve_gemma4_26b_a4b.sh"
echo "Tail logs: tail -f $LOG_DIR/gpu0.log"

cleanup() {
  echo
  echo "==> Shutting down $NUM_INSTANCES vLLM instance(s)..."
  stop_instances "$NUM_INSTANCES"
}
trap cleanup EXIT INT TERM

while true; do
  alive=0
  for ((i = 0; i < NUM_INSTANCES; i++)); do
    pid_file="$PID_DIR/gpu${i}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  if [[ "$alive" -eq 0 ]]; then
    echo "ERROR: all vLLM instances exited; check $LOG_DIR/*.log" >&2
    exit 1
  fi
  sleep 10
done
