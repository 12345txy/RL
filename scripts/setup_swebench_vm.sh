#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_swebench_vm.sh

Prepare a CPU VM for SWE-bench with local Docker (agent + harness eval).
Does NOT install Docker itself — run the Docker steps in docs/swebench_vm_docker_guide.md first.

Optional env:
  CONDA_ENV=swebench   Default conda env name to create
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

CONDA_ENV="${CONDA_ENV:-swebench}"

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker CLI not found. Install Docker first (see docs/swebench_vm_docker_guide.md)." >&2
  exit 1
fi

if ! docker ps >/dev/null 2>&1; then
  echo "ERROR: cannot talk to Docker daemon. Start dockerd and ensure your user is in group 'docker'." >&2
  echo "  sudo usermod -aG docker \$USER && newgrp docker" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found. Install Miniconda or use venv (see guide)." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "==> Using existing conda env: $CONDA_ENV"
else
  echo "==> Creating conda env: $CONDA_ENV (python 3.10)"
  conda create -y -n "$CONDA_ENV" python=3.10
fi
conda activate "$CONDA_ENV"

python -m pip install -U pip
python -m pip install -r requirements-swebench-docker.txt

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"

echo
echo "==> Setup complete (env: $CONDA_ENV)"
echo "    docker: $(docker --version)"
echo "    mini-extra: $(command -v mini-extra)"
echo "    swebench: $(python -c 'import swebench; print(swebench.__file__)')"
echo
echo "Next:"
echo "  1) On GPU machine: bash scripts/serve_gemma4_12b.sh"
echo "  2) SSH tunnel (on VM, if vLLM is not public):"
echo "       ssh -N -L 8000:127.0.0.1:8000 user@GPU_HOST"
echo "  3) Run agent:"
echo "       bash scripts/run_swebench_vm_docker.sh"
echo "  4) Local eval:"
echo "       bash scripts/eval_swebench_local.sh results/swebench_vm_docker/preds.json vm-run-1"
