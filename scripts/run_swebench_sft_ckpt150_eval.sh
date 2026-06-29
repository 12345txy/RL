#!/usr/bin/env bash
# Evaluate SFT checkpoint-150 (LoRA) on SWE-bench Verified with a fixed random 30-instance split.
#
# SSH topology (matches ~/.ssh/config work.bj11 + cpu-mechine-1):
#
#   [GPU work.bj11]  vLLM :8001
#        ^ LocalForward 8001 (ssh work.bj11)
#   [你的 PC]        localhost:8001
#        ^ RemoteForward 8001 (ssh cpu-mechine-1)
#   [CPU cpu-mechine-1]  curl http://127.0.0.1:8001/v1/models
#
# Step 0 — 本机同时保持两条 SSH 连接（顺序：先 GPU，再 CPU）:
#   ssh work.bj11          # LocalForward: 6379,8265,8001,9000
#   ssh cpu-mechine-1      # RemoteForward: 6379,8001,9000 → 本机
#
# Step 1 — GPU (work.bj11)，单独终端，勿与 GRPO 训练同时占用 :8001:
#   cd ~/RL && conda activate RL
#   LORA_PATH=outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150 \
#   PORT=8001 bash scripts/serve_gemma4_12b.sh
#   # LoRA 暴露为 model id "checkpoint-150"
#
# Step 2 — CPU (cpu-mechine-1)，先验连通:
#   curl -s http://127.0.0.1:8001/v1/models | head
#
# Step 3 — CPU 跑 30 题评测:
#   cd ~/RL && conda activate swebench
#   bash scripts/run_swebench_sft_ckpt150_eval.sh
#
# Agent only (skip harness):
#   RUN_HARNESS=0 bash scripts/run_swebench_sft_ckpt150_eval.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SPLIT_FILE="${SPLIT_FILE:-data/splits/verified_eval30_seed42.json}"
PREPARE_SPLIT="${PREPARE_SPLIT:-0}"
SPLIT_N="${SPLIT_N:-30}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SUBSET="${SUBSET:-verified}"
SPLIT="${SPLIT:-test}"
WORKERS="${WORKERS:-2}"
REDO_EXISTING="${REDO_EXISTING:-1}"
RUN_HARNESS="${RUN_HARNESS:-1}"
HARNESS_WORKERS="${HARNESS_WORKERS:-4}"
RUN_ID="${RUN_ID:-sft-ckpt150-eval30-seed42}"

VLLM_BASE="${VLLM_BASE:-http://127.0.0.1:8001/v1}"
MODEL="${MODEL:-hosted_vllm/checkpoint-150}"
CONFIG="${CONFIG:-configs/swebench_docker_gemma4_12b.yaml}"
CONDA_ENV="${CONDA_ENV:-swebench}"
OUTPUT_DIR="${OUTPUT_DIR:-results/swebench_sft_ckpt150_eval30}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_swebench_sft_ckpt150_eval.sh

Run mini-swe-agent on 30 randomly sampled SWE-bench Verified instances, then optional harness eval.

Prerequisites:
  - GPU (work.bj11): vLLM on :8001 with SFT LoRA (see script header)
  - 本机: ssh work.bj11 + ssh cpu-mechine-1 两条隧道同时在线
  - CPU (cpu-mechine-1): Docker + swebench conda env

Env:
  SPLIT_FILE=data/splits/verified_eval30_seed42.json
  PREPARE_SPLIT=1          Regenerate split JSON before running
  SPLIT_N=30 SPLIT_SEED=42
  VLLM_BASE=http://127.0.0.1:8001/v1   # CPU 经 RemoteForward 连 GPU vLLM
  MODEL=hosted_vllm/checkpoint-150
  WORKERS=2
  OUTPUT_DIR=results/swebench_sft_ckpt150_eval30
  RUN_HARNESS=1            Run scripts/eval_swebench_local.sh after agent
  RUN_ID=sft-ckpt150-eval30-seed42
  REDO_EXISTING=1
  MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=2   LLM: 8min timeout, 1 retry (see CONFIG)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v docker >/dev/null 2>&1 || ! docker ps >/dev/null 2>&1; then
  echo "ERROR: Docker is not usable on this machine." >&2
  echo "Run this script on the CPU machine with Docker (see docs/swebench_setup_report.md)." >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if ! command -v mini-extra >/dev/null 2>&1; then
  echo "mini-extra not found. Run: bash scripts/setup_swebench_vm.sh" >&2
  exit 1
fi

if [[ "$PREPARE_SPLIT" == "1" ]]; then
  echo "==> Preparing split: n=$SPLIT_N seed=$SPLIT_SEED -> $SPLIT_FILE"
  python3 - <<PY
import sys
sys.path.insert(0, "$ROOT")
from data.swe_utils import make_verified_dev_split
make_verified_dev_split(n=$SPLIT_N, seed=$SPLIT_SEED, output_path="$SPLIT_FILE")
PY
fi

if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "Split file not found: $SPLIT_FILE" >&2
  echo "Run with PREPARE_SPLIT=1 or copy data/splits/verified_eval30_seed42.json from GPU repo." >&2
  exit 1
fi

FILTER_SPEC="$(python3 - <<PY
import json, re
from pathlib import Path
data = json.loads(Path("$SPLIT_FILE").read_text())
ids = data["instance_ids"] if isinstance(data, dict) else data
if not ids:
    raise SystemExit("empty instance_ids in split file")
print("(" + "|".join(re.escape(i) for i in ids) + ")")
PY
)"

if ! curl -sf "${VLLM_BASE%/}/models" >/dev/null 2>&1; then
  echo "ERROR: vLLM not reachable at ${VLLM_BASE}" >&2
  echo "Checklist:" >&2
  echo "  1) GPU (work.bj11): LORA_PATH=outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150 PORT=8001 bash scripts/serve_gemma4_12b.sh" >&2
  echo "  2) 本机保持 ssh work.bj11（LocalForward 8001）" >&2
  echo "  3) 本机保持 ssh cpu-mechine-1（RemoteForward 8001）" >&2
  echo "  4) CPU 上: curl -s http://127.0.0.1:8001/v1/models" >&2
  exit 1
fi

export MSWEA_COST_TRACKING="${MSWEA_COST_TRACKING:-ignore_errors}"
export MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT="${MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT:-2}"

mkdir -p "$OUTPUT_DIR"

echo "==> SWE-bench Verified eval (SFT checkpoint-150, n=30, sticky routing)"
echo "    split=$SPLIT_FILE"
echo "    model=$MODEL vllm=$VLLM_BASE workers=$WORKERS"
echo "    output=$OUTPUT_DIR"

REDO_ARG=()
if [[ "$REDO_EXISTING" == "1" ]]; then
  REDO_ARG=(--redo-existing)
fi

PYTHONPATH="$ROOT" python -m integrations.miniswe.run_swebench \
  -c swebench \
  -c "$CONFIG" \
  -c "model.model_kwargs.api_base=${VLLM_BASE}" \
  -c "model.model_name=${MODEL#hosted_vllm/}" \
  --environment-class docker \
  --subset "$SUBSET" \
  --split "$SPLIT" \
  --filter "$FILTER_SPEC" \
  --workers "$WORKERS" \
  --model "$MODEL" \
  -o "$OUTPUT_DIR" \
  "${REDO_ARG[@]}"

PREDS="$OUTPUT_DIR/preds.json"
if [[ ! -f "$PREDS" ]]; then
  echo "ERROR: preds.json not found under $OUTPUT_DIR" >&2
  exit 1
fi

python3 - <<PY
import json
from pathlib import Path
preds = json.loads(Path("$PREDS").read_text())
empty = sum(1 for v in preds.values() if not (v.get("model_patch") or "").strip())
print(f"==> Agent done: {len(preds)} preds, {empty} empty patches")
PY

if [[ "$RUN_HARNESS" == "1" ]]; then
  echo
  echo "==> Running SWE-bench harness eval..."
  MAX_WORKERS="$HARNESS_WORKERS" REPORT_DIR="${OUTPUT_DIR}/harness_reports" \
    bash scripts/eval_swebench_local.sh "$PREDS" "$RUN_ID"

  python3 - <<PY
import json
from pathlib import Path
report = Path("logs/run_evaluation/${RUN_ID}")
if not report.exists():
    print("Harness report dir not found:", report)
    raise SystemExit(0)
resolved = 0
total = 0
for p in report.glob("*.json"):
    if p.name in {"report.json", "run_instance.log"}:
        continue
    try:
        row = json.loads(p.read_text())
    except Exception:
        continue
    total += 1
    if row.get("resolved"):
        resolved += 1
print(f"==> Harness summary (approx): resolved={resolved}/{total}")
PY
fi

echo
echo "==> Done."
echo "    preds: $PREDS"
echo "    trajectories: $OUTPUT_DIR/<instance_id>/"
