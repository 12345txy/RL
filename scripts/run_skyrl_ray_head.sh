#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
NUM_GPUS="${NUM_GPUS:-8}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
# Default on: GPU runs in an isolated container; CPU workers join via SSH tunnel to 127.0.0.1.
RAY_TUNNEL_MODE="${RAY_TUNNEL_MODE:-1}"
RAY_TUNNEL_IP="${RAY_TUNNEL_IP:-127.0.0.1}"
RAY_START_HOOK="${RAY_START_HOOK:-integrations.skyrl_miniswe.ray_tunnel_hook.patch_ray_params_for_ssh_tunnel}"
HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"
if [[ "$RAY_TUNNEL_MODE" == "1" ]]; then
  HEAD_IP="$RAY_TUNNEL_IP"
fi

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export RAY_RUNTIME_ENV_HOOK="${RAY_RUNTIME_ENV_HOOK:-ray._private.runtime_env.uv_runtime_env_hook.hook}"
export RAY_TUNNEL_MODE
export RAY_START_HOOK
if [[ "$RAY_TUNNEL_MODE" == "1" ]]; then
  export RAY_PRESERVE_LOCALHOST_IP=1
  python "$ROOT/scripts/patch_ray_tunnel.py"
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyrl_ray_head.sh

Start Ray head on the GPU machine (8×H100).

Env:
  CONDA_ENV=RL
  NUM_GPUS=8
  RAY_PORT=6379
  RAY_TUNNEL_MODE=1          Keep head address at 127.0.0.1 for SSH-tunneled workers
  RAY_TUNNEL_IP=127.0.0.1
  RAY_TUNNEL_MODE=0          Use container NIC IP (direct LAN only)

After this, on the CPU machine (SSH tunnel to 127.0.0.1:6379):
  CONDA_ENV=swebench RAY_ADDRESS=127.0.0.1:6379 bash scripts/run_skyrl_ray_worker.sh
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

RAY_START_ARGS=(--head --port="$RAY_PORT" --dashboard-host=0.0.0.0 --dashboard-port="$DASHBOARD_PORT" --num-gpus="$NUM_GPUS")
if [[ "$RAY_TUNNEL_MODE" == "1" ]]; then
  RAY_START_ARGS+=(--node-ip-address="$RAY_TUNNEL_IP")
  echo "==> Starting Ray head (tunnel mode, advertise $RAY_TUNNEL_IP, gpus=$NUM_GPUS)"
else
  echo "==> Starting Ray head on $HEAD_IP (gpus=$NUM_GPUS)"
fi

ray start "${RAY_START_ARGS[@]}"

echo
echo "==> Ray head ready"
echo "    address: $HEAD_IP:$RAY_PORT"
echo "    dashboard: http://$HEAD_IP:$DASHBOARD_PORT"
if [[ "$RAY_TUNNEL_MODE" == "1" ]]; then
  echo "    tunnel hook: $RAY_START_HOOK"
fi
echo
echo "On CPU VM (Docker, via SSH tunnel):"
echo "  CONDA_ENV=swebench RAY_ADDRESS=$HEAD_IP:$RAY_PORT SKYRL_REQUIRE_DOCKER_NODE=1 bash scripts/run_skyrl_ray_worker.sh"
echo
echo "Then on GPU machine:"
echo "  SKYRL_HTTP_HOST=$HEAD_IP SKYRL_REQUIRE_DOCKER_NODE=1 STAGE=rl1 bash scripts/run_rl_skyrl.sh"
