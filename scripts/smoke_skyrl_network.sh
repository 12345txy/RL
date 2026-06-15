#!/usr/bin/env bash
# Smoke test: SkyRL env + Ray head + vLLM API (no full RL training).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
HTTP_PORT="${SKYRL_HTTP_PORT:-8001}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

PASS=0
FAIL=0
ok() { echo "  [OK] $*"; PASS=$((PASS + 1)); }
bad() { echo "  [FAIL] $*"; FAIL=$((FAIL + 1)); }

echo "==> 1. Python / SkyRL imports"
if python - <<'PY' 2>/dev/null; then
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine
from integrations.skyrl_miniswe.main import BasePPOExp
print("imports ok")
PY
  ok "skyrl + vllm weight-sync APIs"
else
  bad "skyrl / vllm import failed"
fi

echo "==> 2. SFT checkpoint"
CKPT="${SFT_CHECKPOINT:-outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150}"
if [[ -f "$CKPT/adapter_config.json" ]]; then
  ok "LoRA checkpoint: $CKPT"
else
  bad "missing $CKPT/adapter_config.json"
fi

echo "==> 3. RL parquet"
if [[ -f data/rl/skyrl_parquet/train_lite.parquet ]]; then
  ok "train_lite.parquet"
else
  bad "missing data/rl/skyrl_parquet/train_lite.parquet (run preprocess or STAGE=rl1 once)"
fi

echo "==> 4. Docker (CPU rollout prerequisite)"
if command -v docker >/dev/null 2>&1 && docker ps >/dev/null 2>&1; then
  ok "docker available on this host"
else
  bad "docker not available here — SWE rollouts must run on a CPU worker with Docker"
fi

echo "==> 5. Ray head"
if ray status >/dev/null 2>&1; then
  ok "Ray already running"
  ray status 2>&1 | head -20
else
  echo "    starting Ray head on $HEAD_IP ..."
  ray start --head --port="$RAY_PORT" --dashboard-host=0.0.0.0 --dashboard-port="$DASHBOARD_PORT" --num-gpus=8 >/dev/null
  if ray status >/dev/null 2>&1; then
    ok "Ray head started"
    ray status 2>&1 | head -20
  else
    bad "Ray head failed to start"
  fi
fi

echo "==> 6. Listening ports (for your local port-forward test)"
for p in "$RAY_PORT" "$DASHBOARD_PORT"; do
  if ss -tln 2>/dev/null | grep -q ":$p "; then
    ok "port $p listening"
  else
    bad "port $p not listening"
  fi
done

echo
echo "==> Network notes for port-forward to your PC"
echo "    GPU internal IP: $HEAD_IP"
echo "    Ray GCS:         $HEAD_IP:$RAY_PORT  (forward this from GPU -> localhost on PC)"
echo "    Ray dashboard:   http://$HEAD_IP:$DASHBOARD_PORT"
echo "    vLLM HTTP:       http://$HEAD_IP:$HTTP_PORT  (only after run_rl_skyrl.sh starts training)"
echo
echo "    From your PC after SSH -L 6379:$HEAD_IP:6379 -L 8265:$HEAD_IP:8265:"
echo "      curl -s http://127.0.0.1:$DASHBOARD_PORT | head -c 200"
echo "      ray status --address=127.0.0.1:$RAY_PORT"
echo
echo "    CPU Ray worker on your PC (needs Docker on PC):"
echo "      RAY_ADDRESS=127.0.0.1:$RAY_PORT bash scripts/run_skyrl_ray_worker.sh"
echo "    Warning: Ray cross-NAT may fail if head advertises $HEAD_IP but worker only sees localhost."
echo

echo "==> Summary: $PASS passed, $FAIL failed"
if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi
