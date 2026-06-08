#!/usr/bin/env bash
# Run lm-evaluation-harness (vLLM backend) for MBPP + HumanEval.
set -euo pipefail

cd "$(dirname "$0")/.."

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export HF_ALLOW_CODE_EVAL=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Qwen3.5 bf16 does not need DeepGEMM; vendored deep_gemm is outdated and crashes warmup.
export VLLM_USE_DEEP_GEMM=0

MODEL_PATH="${1:-models/gemma-4-E4B-it}"
OUTPUT_PATH="${2:-results/lm_eval}"
ADAPTER_PATH="${3:-}"
LIMIT="${4:-}"
# Qwen3.5 reports 262144 max context; cap for MBPP 3-shot (~4.3k tokens) + generation.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.85}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,trust_remote_code=True,max_model_len=${MAX_MODEL_LEN},tensor_parallel_size=${TENSOR_PARALLEL_SIZE},gpu_memory_utilization=${GPU_MEMORY_UTIL},enable_thinking=False"
if [[ -n "${ADAPTER_PATH}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},lora_local_path=${ADAPTER_PATH},max_lora_rank=32"
fi

LIMIT_ARG=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

echo "==> lm-eval (vllm): model=${MODEL_PATH} adapter=${ADAPTER_PATH:-none} output=${OUTPUT_PATH}"

lm_eval run \
  --model vllm \
  --model_args "${MODEL_ARGS}" \
  --tasks mbpp_instruct,humaneval_instruct \
  --apply_chat_template \
  --batch_size "${BATCH_SIZE}" \
  --confirm_run_unsafe_code \
  --output_path "${OUTPUT_PATH}" \
  --log_samples \
  "${LIMIT_ARG[@]}"

python scripts/summarize_lm_eval.py --results_dir "${OUTPUT_PATH}"
