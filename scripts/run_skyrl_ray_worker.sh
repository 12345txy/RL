#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-skyrl}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"
DOCKER_RAY_RESOURCE="${SKYRL_DOCKER_RAY_RESOURCE:-docker_node}"

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export RAY_RUNTIME_ENV_HOOK="${RAY_RUNTIME_ENV_HOOK:-ray._private.runtime_env.uv_runtime_env_hook.hook}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_skyrl_ray_worker.sh

Join a CPU VM (with Docker) to the Ray cluster for Mini-SWE-Agent rollouts.

Prerequisites on this machine:
  - Docker installed and usable (docker ps)
  - bash scripts/setup_skyrl.sh  (or at least: pip install ray mini-swe-agent)

Env:
  RAY_ADDRESS=<GPU_HEAD_IP>:6379   required
  NUM_CPUS=auto
  SKYRL_DOCKER_RAY_RESOURCE=docker_node
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$RAY_ADDRESS" ]]; then
  echo "ERROR: set RAY_ADDRESS to the GPU Ray head, e.g. 10.0.0.1:6379" >&2
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

echo "==> Joining Ray cluster at $RAY_ADDRESS (cpus=$NUM_CPUS, resource=$DOCKER_RAY_RESOURCE=1)"
ray start --address="$RAY_ADDRESS" \
  --num-cpus="$NUM_CPUS" \
  --num-gpus=0 \
  --resources="{\"$DOCKER_RAY_RESOURCE\": 1}"

echo "==> CPU Ray worker ready. Docker rollouts will schedule here when SKYRL_REQUIRE_DOCKER_NODE=1"
