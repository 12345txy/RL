#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKYRL_ENV="${SKYRL_ENV:-RL}"
SKYRL_REPO="${SKYRL_REPO:-$ROOT/vendor/SkyRL}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_skyrl.sh

Install patched SkyRL into the shared RL conda env (reuses existing vLLM nightly + torch).

Does NOT downgrade vllm/torch — required for Gemma4-12B Unified + weight sync.

Patches in vendor/SkyRL:
  - lora.adapter_path: init RL from SFT PEFT checkpoint
  - fsdp extra: no vllm/torch pins (use RL env versions)
  - ref inherits policy LoRA for KL

Env:
  SKYRL_ENV=RL
  SKYRL_REPO=vendor/SkyRL
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -d "$SKYRL_REPO/skyrl" ]]; then
  echo "ERROR: missing $SKYRL_REPO (patched SkyRL vendor tree)" >&2
  exit 1
fi

if ! grep -q "adapter_path" "$SKYRL_REPO/skyrl/train/config/config.py"; then
  echo "ERROR: vendor/SkyRL missing SFT adapter patches" >&2
  exit 1
fi

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$SKYRL_ENV"

unset ALL_PROXY HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY 2>/dev/null || true

echo "==> Installing patched SkyRL into conda env: $SKYRL_ENV (python: $(python -V))"
python -m pip install -U "pip" "setuptools<82" wheel

# SkyRL core (editable) without pulling pinned vllm/torch from fsdp extra
python -m pip install -e "$SKYRL_REPO" --no-deps
python -m pip install -e "$SKYRL_REPO/skyrl-gym" --no-deps 2>/dev/null || true

# Remaining SkyRL runtime deps (except vllm/torch — keep env versions)
python -m pip install -r "$ROOT/requirements-skyrl-rl.txt"

export RAY_RUNTIME_ENV_HOOK="${RAY_RUNTIME_ENV_HOOK:-ray._private.runtime_env.uv_runtime_env_hook.hook}"

python - <<'PY'
import skyrl, torch, vllm, transformers, ray
from vllm.config import WeightTransferConfig
from skyrl.backends.skyrl_train.inference_servers.utils import build_vllm_cli_args
print("skyrl:", skyrl.__file__)
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("ray:", ray.__version__)
print("WeightTransferConfig: OK")
print("skyrl.backends: OK")
PY

echo
echo "==> Patched SkyRL ready in conda env: $SKYRL_ENV"
echo "    export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook"
echo "    STAGE=rl1 bash scripts/run_rl_skyrl.sh"
