#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

PREDS_PATH="${1:-}"
RUN_ID="${2:-}"
SUBSET="${SUBSET:-swe-bench_verified}"
SPLIT="${SPLIT:-test}"
REPORT_DIR="${REPORT_DIR:-results/swebench_sbcli_reports}"

usage() {
  cat <<'EOF'
Usage: bash scripts/eval_swebench_sbcli.sh <preds.json> <run_id>

Submit predictions to SWE-bench cloud evaluation (sb-cli).

Args:
  preds.json   Path to preds.json from run_swebench_cloud.sh
  run_id       Unique run name (e.g. qwen35-2b-grpo-verified-5)

Env:
  SWEBENCH_API_KEY   Required API key (see setup_swebench_cloud.sh)
  SUBSET             Default: swe-bench_verified
  SPLIT              Default: test
  REPORT_DIR         Default: results/swebench_sbcli_reports

After submit (~20 min), fetch report:
  sb-cli get-report swe-bench_verified test <run_id> -o results/swebench_sbcli_reports
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

if ! command -v sb-cli >/dev/null 2>&1; then
  echo "sb-cli not found. Run: bash scripts/setup_swebench_cloud.sh" >&2
  exit 1
fi

if [[ -z "${SWEBENCH_API_KEY:-}" ]]; then
  echo "SWEBENCH_API_KEY is not set." >&2
  echo "Run: sb-cli gen-api-key your.email@example.com" >&2
  echo "Then: export SWEBENCH_API_KEY=<your_key>" >&2
  exit 1
fi

mkdir -p "$REPORT_DIR"

echo "==> Submitting to sb-cli"
echo "    subset=$SUBSET split=$SPLIT run_id=$RUN_ID"
echo "    preds=$PREDS_PATH"

sb-cli submit "$SUBSET" "$SPLIT" \
  --predictions_path "$PREDS_PATH" \
  --run_id "$RUN_ID" \
  --output_dir "$REPORT_DIR" \
  --gen_report 1

echo
echo "==> Submitted. Report will be saved under $REPORT_DIR when ready."
echo "    Check status: sb-cli list-runs $SUBSET $SPLIT"
echo "    Fetch report: sb-cli get-report $SUBSET $SPLIT $RUN_ID -o $REPORT_DIR"
