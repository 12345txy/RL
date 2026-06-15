#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-swebench}"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"
DOCKER_RAY_RESOURCE="${SKYRL_DOCKER_RAY_RESOURCE:-docker_node}"
RAY_TUNNEL_MODE="${RAY_TUNNEL_MODE:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export RAY_RUNTIME_ENV_HOOK="${RAY_RUNTIME_ENV_HOOK:-ray._private.runtime_env.uv_runtime_env_hook.hook}"
if [[ "$RAY_TUNNEL_MODE" == "1" ]]; then
  export RAY_PRESERVE_LOCALHOST_IP=1
  python "$ROOT/scripts/patch_ray_tunnel.py"
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyrl_ray_worker.sh

Join a CPU VM (with Docker) to the Ray cluster for Mini-SWE-Agent rollouts.

Prerequisites on this machine:
  - Docker installed and usable (docker ps)
  - pip install "ray[default]" mini-swe-agent  (swebench env)

Env:
  CONDA_ENV=swebench
  RAY_ADDRESS=127.0.0.1:6379   required with SSH tunnel + GPU run_skyrl_ray_head.sh
  RAY_TUNNEL_MODE=1            Patch Ray client to keep 127.0.0.1 (default)
  NUM_CPUS=auto
  SKYRL_DOCKER_RAY_RESOURCE=docker_node
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$RAY_ADDRESS" ]]; then
  echo "ERROR: set RAY_ADDRESS to the GPU Ray head, e.g. 127.0.0.1:6379" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found. Install Docker on this CPU VM first." >&2
  exit 1
fi

if ! docker ps >/dev/null 2>&1; then
  echo "ERROR: cannot talk to Docker daemon." >&2
  exit 1
fi

DOCKER_BIN="$(command -v docker)"
export DOCKER_EXECUTABLE="$DOCKER_BIN"
export MSWEA_DOCKER_EXECUTABLE="$DOCKER_BIN"
export PATH="$(dirname "$DOCKER_BIN"):${PATH}"
# Ray worker subprocesses inherit the raylet env; pin docker for Mini-SWE-Agent.
export SKYRL_DOCKER_EXECUTABLE="$DOCKER_BIN"

echo "==> Joining Ray cluster at $RAY_ADDRESS (cpus=$NUM_CPUS, resource=$DOCKER_RAY_RESOURCE=1, docker=$DOCKER_BIN)"
ray start --address="$RAY_ADDRESS" \
  --num-cpus="$NUM_CPUS" \
  --num-gpus=0 \
  --resources="{\"$DOCKER_RAY_RESOURCE\": 1}"

echo "==> CPU Ray worker ready. Docker rollouts will schedule here when SKYRL_REQUIRE_DOCKER_NODE=1"
