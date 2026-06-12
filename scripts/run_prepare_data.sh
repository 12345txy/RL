#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
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

Phase 1: build mini-swe-agent SFT + SWE-Gym RL datasets with Verified leakage filtering.

Env:
  MINISWE_HF=Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k
  MINISWE_EXTRA_HF=JetBrains-Research/agent-trajectories-swesmith-random-subset
  MINISWE_EXTRA_HF=                      Skip supplemental JetBrains set
  MINISWE_MAX_SAMPLES=5000
  MINISWE_EXTRA_MAX_SAMPLES=1500
  MINISWE_TRAJ=data/raw/miniswe_trajectories   Optional local traj dir/jsonl

Outputs:
  data/sft/miniswe_train.jsonl
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

echo "==> mini-swe-agent SFT (primary=$MINISWE_HF)"
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

echo "==> Merge SFT datasets (mini-swe only)"
python data/merge_sft_datasets.py \
  --inputs data/sft/miniswe_train.jsonl \
  --output data/sft/sft_merged.jsonl

echo "==> Data prep complete"
