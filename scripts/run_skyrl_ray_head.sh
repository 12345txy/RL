#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-skyrl}"
NUM_GPUS="${NUM_GPUS:-8}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export RAY_RUNTIME_ENV_HOOK="${RAY_RUNTIME_ENV_HOOK:-ray._private.runtime_env.uv_runtime_env_hook.hook}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyrl_ray_head.sh

Start Ray head on the GPU machine (8×H100).

Env:
  NUM_GPUS=8
  HEAD_IP=auto-detected
  RAY_PORT=6379

After this, on the CPU machine:
  RAY_ADDRESS=<HEAD_IP>:6379 bash scripts/run_skyrl_ray_worker.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ray status >/dev/null 2>&1; then
  echo "==> Ray already running"
  ray status
  exit 0
fi

echo "==> Starting Ray head on $HEAD_IP (gpus=$NUM_GPUS)"
ray start --head \
  --port="$RAY_PORT" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port="$DASHBOARD_PORT" \
  --num-gpus="$NUM_GPUS"

echo
echo "==> Ray head ready"
echo "    address: $HEAD_IP:$RAY_PORT"
echo "    dashboard: http://$HEAD_IP:$DASHBOARD_PORT"
echo
echo "On CPU VM (Docker):"
echo "  RAY_ADDRESS=$HEAD_IP:$RAY_PORT SKYRL_REQUIRE_DOCKER_NODE=1 bash scripts/run_skyrl_ray_worker.sh"
echo
echo "Then on GPU machine:"
echo "  SKYRL_HTTP_HOST=$HEAD_IP SKYRL_REQUIRE_DOCKER_NODE=1 STAGE=rl1 bash scripts/run_rl_skyrl.sh"
