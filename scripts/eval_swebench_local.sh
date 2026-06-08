#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PREDS_PATH="${1:-}"
RUN_ID="${2:-}"
DATASET="${DATASET:-princeton-nlp/SWE-bench_Verified}"
SPLIT="${SPLIT:-test}"
MAX_WORKERS="${MAX_WORKERS:-4}"
REPORT_DIR="${REPORT_DIR:-results/swebench_local_reports}"
CONDA_ENV="${CONDA_ENV:-swebench}"
CLEAN="${CLEAN:-0}"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_swebench_local.sh <preds.json> <run_id>

Run SWE-bench harness evaluation locally in Docker (no sb-cli).

Args:
  preds.json   Predictions from run_swebench_vm_docker.sh (or copied from GPU/Modal run)
  run_id       Unique run name for report directory

Env:
  MAX_WORKERS=4
  REPORT_DIR=results/swebench_local_reports
  CLEAN=1              Pass --clean to harness (re-run all instances)
  CONDA_ENV=swebench

Example:
  bash scripts/eval_swebench_local.sh results/swebench_vm_docker/preds.json vm-docker-100
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ -z "$PREDS_PATH" || -z "$RUN_ID" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$PREDS_PATH" ]]; then
  echo "Predictions file not found: $PREDS_PATH" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1 || ! docker ps >/dev/null 2>&1; then
  echo "ERROR: Docker is not usable." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python - <<'PY' "$PREDS_PATH"
import json, sys
path = sys.argv[1]
data = json.load(open(path))
if isinstance(data, dict):
    n = len(data)
    empty = sum(1 for v in data.values() if not (v.get("model_patch") or "").strip())
    print(f"preds.json: {n} instances, {empty} empty patches")
PY

mkdir -p "$REPORT_DIR"
LOG="$REPORT_DIR/${RUN_ID}.log"

CLEAN_ARG=()
if [[ "$CLEAN" == "1" ]]; then
  CLEAN_ARG=(--clean)
fi

echo "==> Local SWE-bench harness eval"
echo "    dataset=$DATASET split=$SPLIT run_id=$RUN_ID"
echo "    preds=$PREDS_PATH max_workers=$MAX_WORKERS"
echo "    log=$LOG"

python -m swebench.harness.run_evaluation \
  --dataset_name "$DATASET" \
  --split "$SPLIT" \
  --predictions_path "$PREDS_PATH" \
  --max_workers "$MAX_WORKERS" \
  --run_id "$RUN_ID" \
  "${CLEAN_ARG[@]}" \
  2>&1 | tee "$LOG"

echo
echo "==> Done. Check log: $LOG"
echo "    Reports under: logs/run_evaluation/$RUN_ID/ (swebench default)"
