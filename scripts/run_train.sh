#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_ALLOW_CODE_EVAL=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export SWANLAB_PROJECT="${SWANLAB_PROJECT:-coding-rl}"
export SWANLAB_MODE="${SWANLAB_MODE:-local}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"
SKIP_DATA_PREP="${SKIP_DATA_PREP:-0}"
USE_FLA="${USE_FLA:-0}"
export USE_FLA
export FLA_TILELANG="${FLA_TILELANG:-0}"

MODEL_PATH="${MODEL_PATH:-models/Qwen3.5-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/grpo-mbpp-qwen35-2b-beta}"
MODEL_NAME="$(basename "${MODEL_PATH}")"
BASELINE_EVAL_DIR="results/lm_eval_baseline/models__${MODEL_NAME}"
TRAINED_EVAL_DIR="results/lm_eval_trained/models__${MODEL_NAME}"
BASELINE_SUMMARY="results/baseline_summary_${MODEL_NAME}.json"
TRAINED_SUMMARY="results/trained_summary_${MODEL_NAME}.json"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

if [[ "${SKIP_DATA_PREP}" != "1" ]]; then
  echo "==> Preparing MBPP dataset (official full splits, lm-eval prompts)"
  python data/prepare_mbpp.py
else
  echo "==> Skipping data prep (SKIP_DATA_PREP=1)"
fi

if [[ "${SKIP_BASELINE}" != "1" ]]; then
  echo "==> Baseline evaluation (lm-eval: mbpp_instruct + humaneval_instruct)"
  bash scripts/eval_lm_eval.sh "${MODEL_PATH}" results/lm_eval_baseline
  cp "$(find results/lm_eval_baseline -path "*models__${MODEL_NAME}*" -name 'results_*.json' | sort | tail -1)" results/baseline_eval.json 2>/dev/null || true
  python scripts/summarize_lm_eval.py --results_dir results/lm_eval_baseline --output "${BASELINE_SUMMARY}"
else
  echo "==> Skipping baseline eval (SKIP_BASELINE=1)"
  if [[ ! -f "${BASELINE_SUMMARY}" ]]; then
    if [[ -d "${BASELINE_EVAL_DIR}" ]]; then
      python scripts/summarize_lm_eval.py --results_dir "${BASELINE_EVAL_DIR}" --output "${BASELINE_SUMMARY}"
    elif [[ -d "results/lm_eval/models__${MODEL_NAME}" ]]; then
      echo "WARN: using results/lm_eval/models__${MODEL_NAME} as baseline fallback"
      python scripts/summarize_lm_eval.py --results_dir "results/lm_eval/models__${MODEL_NAME}" --output "${BASELINE_SUMMARY}"
    else
      echo "WARN: ${BASELINE_SUMMARY} missing; before/after summary may be incomplete"
    fi
  fi
fi

echo "==> GRPO training"
accelerate launch --config_file configs/accelerate_2gpu.yaml scripts/train_grpo.py \
  --model_path "${MODEL_PATH}" \
  --train_path data/processed/mbpp_train \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 2 \
  --num_generations 8 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --generation_batch_size 128 \
  --beta 0.02 \
  --learning_rate 5e-6

echo "==> Post-training evaluation (lm-eval)"
bash scripts/eval_lm_eval.sh "${MODEL_PATH}" results/lm_eval_trained "${OUTPUT_DIR}"
python scripts/summarize_lm_eval.py --results_dir results/lm_eval_trained --output "${TRAINED_SUMMARY}"

python - <<PY
import json
from pathlib import Path

model_name = "${MODEL_NAME}"
baseline_path = Path("${BASELINE_SUMMARY}")
trained_path = Path("${TRAINED_SUMMARY}")

def load_summary(path):
    if not path.exists():
        return {}, ""
    data = json.loads(path.read_text())
    return {m["task"]: m["pass@1"] for m in data["metrics"]}, data.get("results_file", "")

base, base_file = load_summary(baseline_path)
trained, trained_file = load_summary(trained_path)

if not base:
    print(f"WARN: missing baseline summary: {baseline_path}")
if model_name not in base_file and base_file:
    print(f"WARN: baseline results may be for a different model: {base_file}")
if model_name not in trained_file and trained_file:
    print(f"WARN: trained results may be for a different model: {trained_file}")

print(f"\n=== Before / After (lm-eval pass@1, model={model_name}) ===")
for task in sorted(set(base) | set(trained)):
    b = base.get(task, float("nan"))
    t = trained.get(task, float("nan"))
    delta = t - b if b == b and t == t else float("nan")
    print(f"{task:22s}  baseline={b:.4f}  trained={t:.4f}  delta={delta:+.4f}")
PY
