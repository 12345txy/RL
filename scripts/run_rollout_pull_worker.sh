#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-swebench}"
QUEUE_URL="${SKYRL_ROLLOUT_QUEUE_URL:-http://127.0.0.1:9000}"
PULL_WORKERS="${SKYRL_ROLLOUT_PULL_WORKERS:-4}"
VLLM_BASE="${OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}"
DEQUEUE_TIMEOUT="${SKYRL_ROLLOUT_DEQUEUE_TIMEOUT_S:-30}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export SKYRL_ROLLOUT_QUEUE_URL="$QUEUE_URL"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"
export OPENAI_BASE_URL="$VLLM_BASE"
export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_rollout_pull_worker.sh

Pull Mini-SWE Docker rollouts from the GPU rollout queue (no Ray worker needed).

Prerequisites on this CPU machine:
  - Docker installed (docker ps)
  - SSH: 本机同时 ssh work.bj11 + ssh cpu-mechine-1
      work.bj11 LocalForward: 6379,8265,8001,9000
      cpu-mechine-1 RemoteForward: 6379,8001,9000
  - GPU training running (queue http://127.0.0.1:9000, vLLM http://127.0.0.1:8001)

Env:
  CONDA_ENV=swebench
  SKYRL_ROLLOUT_QUEUE_URL=http://127.0.0.1:9000
  SKYRL_ROLLOUT_PULL_WORKERS=4
  OPENAI_BASE_URL=http://127.0.0.1:8001/v1   # SkyRL vLLM via SSH tunnel
  SKYRL_ROLLOUT_DEQUEUE_TIMEOUT_S=30
  MSWEA_COST_TRACKING=ignore_errors   # local vLLM model not in LiteLLM price table
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found." >&2
  exit 1
fi

if ! docker ps >/dev/null 2>&1; then
  echo "ERROR: cannot talk to Docker daemon." >&2
  exit 1
fi

DOCKER_BIN="$(command -v docker)"
export SKYRL_DOCKER_EXECUTABLE="$DOCKER_BIN"
export MSWEA_DOCKER_EXECUTABLE="$DOCKER_BIN"
export DOCKER_EXECUTABLE="$DOCKER_BIN"

echo "==> Pull rollout worker: queue=$QUEUE_URL workers=$PULL_WORKERS vllm=$VLLM_BASE docker=$DOCKER_BIN"
exec python -m integrations.skyrl_miniswe.rollout_pull_worker_main \
  --queue-url "$QUEUE_URL" \
  --workers "$PULL_WORKERS" \
  --dequeue-timeout "$DEQUEUE_TIMEOUT"
