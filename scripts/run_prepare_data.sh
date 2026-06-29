#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
REBENCH_NATIVE_DIR="${REBENCH_NATIVE_DIR:-rebench_native_tool_clean_resolved_repaired_225}"
REBENCH_MAX_CHARS="${REBENCH_MAX_CHARS:-0}"
REBENCH_FILTER_MAX_TOKENS="${REBENCH_FILTER_MAX_TOKENS:-200000}"
REBENCH_FILTER_MODEL="${REBENCH_FILTER_MODEL:-models/gemma-4-12B-it}"
USE_LEGACY_MINISWE="${USE_LEGACY_MINISWE:-0}"
MINISWE_HF="${MINISWE_HF:-Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k}"
MINISWE_SPLIT="${MINISWE_SPLIT:-train}"
MINISWE_EXTRA_HF="${MINISWE_EXTRA_HF:-JetBrains-Research/agent-trajectories-swesmith-random-subset}"
MINISWE_EXTRA_SPLIT="${MINISWE_EXTRA_SPLIT:-train}"
MINISWE_TRAJ="${MINISWE_TRAJ:-}"
MINISWE_MAX_SAMPLES="${MINISWE_MAX_SAMPLES:-5000}"
MINISWE_EXTRA_MAX_SAMPLES="${MINISWE_EXTRA_MAX_SAMPLES:-1500}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_prepare_data.sh

Phase 1: build native-tool SFT + SWE-Gym RL datasets with Verified leakage filtering.

Default SFT source:
  rebench_native_tool_clean_resolved_repaired_225/records.jsonl

Env:
  REBENCH_NATIVE_DIR=rebench_native_tool_clean_resolved_repaired_225
  REBENCH_MAX_CHARS=0                 0 = keep all rows (truncate at train time)
  REBENCH_FILTER_MAX_TOKENS=200000    Drop extreme outliers before SFT
  REBENCH_FILTER_MODEL=models/gemma-4-12B-it
  USE_LEGACY_MINISWE=0                Set 1 to also pull legacy THOUGHT/ACTION HF data
  MINISWE_HF=...                      Used only when USE_LEGACY_MINISWE=1
  MINISWE_EXTRA_HF=...
  MINISWE_MAX_SAMPLES=5000
  MINISWE_EXTRA_MAX_SAMPLES=1500
  MINISWE_TRAJ=                       Optional local traj dir/jsonl for legacy mini-swe

Outputs:
  data/sft/rebench_native_train.jsonl
  data/sft/sft_merged.jsonl
  data/rl/swegym_rl_train.jsonl
  data/splits/verified_dev_100.json
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export HF_ENDPOINT

mkdir -p data/sft data/rl data/splits data/raw

echo "==> Verified dev split"
python - <<'PY'
from data.swe_utils import make_verified_dev_split
make_verified_dev_split(n=100, output_path="data/splits/verified_dev_100.json")
print("data/splits/verified_dev_100.json")
PY

echo "==> SWE-Gym RL pool"
python data/prepare_swe_gym_sft.py --rl_output data/rl/swegym_rl_train.jsonl --rl_only

echo "==> SWE-ReBench native-tool SFT ($REBENCH_NATIVE_DIR)"
python data/prepare_rebench_native_sft.py \
  --records_path "$REBENCH_NATIVE_DIR/records.jsonl" \
  --train_path "$REBENCH_NATIVE_DIR/train.jsonl" \
  --output data/sft/rebench_native_train.jsonl \
  --stats_output data/sft/rebench_native_stats.json \
  --max_chars "$REBENCH_MAX_CHARS" \
  --filter_max_tokens "$REBENCH_FILTER_MAX_TOKENS" \
  --model_path "$REBENCH_FILTER_MODEL"

MERGE_INPUTS=(data/sft/rebench_native_train.jsonl)

if [[ "$USE_LEGACY_MINISWE" == "1" ]]; then
  echo "==> Legacy mini-swe-agent THOUGHT/ACTION SFT (USE_LEGACY_MINISWE=1)"
  MINISWE_ARGS=(
    --output data/sft/miniswe_train.jsonl
    --stats_output data/sft/miniswe_stats.json
    --max_samples "$MINISWE_MAX_SAMPLES"
    --extra_max_samples "$MINISWE_EXTRA_MAX_SAMPLES"
  )
  if [[ -n "$MINISWE_TRAJ" && -e "$MINISWE_TRAJ" ]]; then
    MINISWE_ARGS+=(--traj_dir "$MINISWE_TRAJ")
  else
    MINISWE_ARGS+=(--hf_dataset "$MINISWE_HF" --hf_split "$MINISWE_SPLIT")
    if [[ -n "$MINISWE_EXTRA_HF" ]]; then
      MINISWE_ARGS+=(--extra_hf_dataset "$MINISWE_EXTRA_HF" --extra_hf_split "$MINISWE_EXTRA_SPLIT")
    else
      MINISWE_ARGS+=(--extra_hf_dataset "")
    fi
  fi
  python data/prepare_miniswe_sft.py "${MINISWE_ARGS[@]}"
  MERGE_INPUTS+=(data/sft/miniswe_train.jsonl)
fi

echo "==> Merge SFT datasets"
python data/merge_sft_datasets.py \
  --inputs "${MERGE_INPUTS[@]}" \
  --output data/sft/sft_merged.jsonl \
  --stats_output data/sft/sft_merged_stats.json

echo "==> Data prep complete"
