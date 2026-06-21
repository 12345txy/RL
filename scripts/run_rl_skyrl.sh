#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-RL}"
STAGE="${STAGE:-rl1}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-models/gemma-4-12B-it}"
SFT_CHECKPOINT="${SFT_CHECKPOINT:-outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/rl-gemma4-12b-miniswe}"
CKPT_PATH="${CKPT_PATH:-$OUTPUT_DIR/checkpoints}"
TRAJ_DIR="${TRAJ_DIR:-$OUTPUT_DIR/trajectories}"
PARQUET_DIR="${PARQUET_DIR:-data/rl/skyrl_parquet}"
MINISWE_CONFIG="${MINISWE_CONFIG:-integrations/skyrl_miniswe/swebench.yaml}"

POLICY_GPUS="${POLICY_GPUS:-6}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"
NUM_ENGINES="${NUM_ENGINES:-2}"
TP_SIZE="${TP_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.70}"
# vLLM "ray" executor spawns EngineCore workers that call ray.init() again; with
# RAY_TUNNEL_MODE the GCS bootstrap address (container NIC) is unreachable and
# servers never pass /health. Use "mp" for TP=1 (local GPU workers, no nested Ray).
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
if [[ -z "$DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
  if [[ "$TP_SIZE" -le 1 ]]; then
    DISTRIBUTED_EXECUTOR_BACKEND=mp
  else
    DISTRIBUTED_EXECUTOR_BACKEND=ray
  fi
fi
HTTP_HOST="${SKYRL_HTTP_HOST:-127.0.0.1}"
HTTP_PORT="${SKYRL_HTTP_PORT:-8001}"
SKYRL_ROLLOUT_MODE="${SKYRL_ROLLOUT_MODE:-pull}"
SKYRL_ROLLOUT_QUEUE_HOST="${SKYRL_ROLLOUT_QUEUE_HOST:-127.0.0.1}"
SKYRL_ROLLOUT_QUEUE_PORT="${SKYRL_ROLLOUT_QUEUE_PORT:-9000}"
SKYRL_ROLLOUT_PULL_WORKERS="${SKYRL_ROLLOUT_PULL_WORKERS:-4}"
SKYRL_REQUIRE_DOCKER_NODE="${SKYRL_REQUIRE_DOCKER_NODE:-0}"
SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-}"
LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-128}"
EPOCHS="${EPOCHS:-}"
LOGGER="${LOGGER:-swanlab}"
PROJECT_NAME="${SWANLAB_PROJECT:-swe-rl}"
SWANLAB_MODE="${SWANLAB_MODE:-local}"
SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-$ROOT/swanlog}"
RUN_NAME="${SWANLAB_EXPERIMENT_NAME:-}"
RAY_TUNNEL_MODE="${RAY_TUNNEL_MODE:-1}"
RAY_TUNNEL_IP="${RAY_TUNNEL_IP:-127.0.0.1}"
# FSDP policy/ref NCCL collective timeout (seconds). Default 300 = 5 minutes.
SKYRL_WORKER_NCCL_TIMEOUT_IN_S="${SKYRL_WORKER_NCCL_TIMEOUT_IN_S:-300}"
# Mini-SWE pull workers call a stable OpenAI-compatible endpoint on SKYRL_HTTP_PORT
# (default 8001). SkyRL's new inference path binds the router to a random port instead,
# so disable it unless you also retarget CPU OPENAI_BASE_URL to that proxy URL.
SKYRL_USE_NEW_INFERENCE="${SKYRL_USE_NEW_INFERENCE:-0}"

source "/root/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export RAY_RUNTIME_ENV_HOOK="${RAY_RUNTIME_ENV_HOOK:-ray._private.runtime_env.uv_runtime_env_hook.hook}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export SKYRL_REQUIRE_DOCKER_NODE
export SKYRL_ROLLOUT_MODE SKYRL_ROLLOUT_QUEUE_HOST SKYRL_ROLLOUT_QUEUE_PORT
export _SKYRL_USE_NEW_INFERENCE="$SKYRL_USE_NEW_INFERENCE"
export SWANLAB_MODE SWANLAB_LOG_DIR
export SKYRL_WORKER_NCCL_TIMEOUT_IN_S
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ "$SWANLAB_MODE" == "local" || "$SWANLAB_MODE" == "disabled" ]]; then
  unset SWANLAB_API_KEY
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
fi
if [[ "$RAY_TUNNEL_MODE" == "1" ]]; then
  export RAY_PRESERVE_LOCALHOST_IP=1
  export RAY_ADDRESS="${RAY_ADDRESS:-$RAY_TUNNEL_IP:6379}"
  python "$ROOT/scripts/patch_ray_tunnel.py"
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/run_rl_skyrl.sh

True SkyRL GRPO training with:
  - FSDP policy update on 6 GPUs
  - vLLM rollout engines on 2 GPUs (NCCL weight sync after each step)
  - Mini-SWE-Agent Docker rollouts via pull queue on CPU (default)

Stages:
  rl1  SWE-Gym-Lite parquet, 5 epochs (default)
  rl2  Full SWE-Gym parquet, 20 epochs

Prerequisites:
  bash scripts/setup_skyrl.sh
  bash scripts/run_skyrl_ray_head.sh        # GPU machine
  bash scripts/run_rollout_pull_worker.sh   # CPU machine (Docker)

Env:
  STAGE=rl1|rl2
  SKYRL_ROLLOUT_MODE=pull|ray             # default pull (SSH tunnel friendly)
  SKYRL_ROLLOUT_QUEUE_PORT=9000           # add SSH forward for 9000 like 8001
  SKYRL_ROLLOUT_PULL_WORKERS=4            # concurrent Docker workers on CPU
  SKYRL_REQUIRE_DOCKER_NODE=0             # set 1 only for legacy Ray rollout mode
  POLICY_MODEL_PATH=models/gemma-4-12B-it
  SFT_CHECKPOINT=outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150
  SFT_ADAPTER_PATH=outputs/.../checkpoint-150   # optional; auto from SFT_CHECKPOINT if LoRA
  LORA_RANK=0|64
  POLICY_GPUS=6  ROLLOUT_GPUS=2  NUM_ENGINES=2
  VLLM_GPU_MEMORY_UTILIZATION=0.70  # lower if NCCL weight sync OOMs/timeouts
  VLLM_ENABLE_AUTO_TOOL_CHOICE=true  # required for mini-swe-agent tool_choice=auto
  VLLM_TOOL_CALL_PARSER=gemma4       # Gemma4 unified tool calling parser
  SKYRL_WORKER_NCCL_TIMEOUT_IN_S=300  # FSDP NCCL collective timeout (5 minutes)
  MAX_TRAIN_SEQ_LEN=18432             # policy backward cap; do NOT set 0
  SKYRL_HTTP_HOST=127.0.0.1   # GPU vLLM HTTP for CPU pull workers (tunnel)
  SKYRL_USE_NEW_INFERENCE=0   # required for stable :8001 OpenAI endpoint (pull rollout)
  SWANLAB_MODE=local           # default: local logs only, no cloud upload
  SWANLAB_LOG_DIR=swanlog
  OUTPUT_DIR=outputs/rl-gemma4-12b-miniswe
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if ! python -c "import skyrl" 2>/dev/null; then
  echo "ERROR: skyrl not installed. Run: bash scripts/setup_skyrl.sh" >&2
  exit 1
fi

LORA_RANK="${LORA_RANK:-0}"

if [[ -d "$SFT_CHECKPOINT" && -f "$SFT_CHECKPOINT/config.json" ]]; then
  POLICY_MODEL_PATH="$SFT_CHECKPOINT"
elif [[ -d "$SFT_CHECKPOINT" && -f "$SFT_CHECKPOINT/adapter_config.json" ]]; then
  SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-$SFT_CHECKPOINT}"
  if [[ "$LORA_RANK" -le 0 ]]; then
    read -r LORA_RANK LORA_ALPHA < <(python - <<'PY' "$SFT_ADAPTER_PATH"
import json, sys
cfg = json.load(open(f"{sys.argv[1]}/adapter_config.json"))
print(cfg.get("r", 64), cfg.get("lora_alpha", 128))
PY
)
  fi
  echo "==> Loading SFT LoRA adapter: $SFT_ADAPTER_PATH (rank=$LORA_RANK alpha=$LORA_ALPHA) on base $POLICY_MODEL_PATH"
elif [[ "$SFT_CHECKPOINT" == outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150 ]]; then
  echo "ERROR: default SFT LoRA checkpoint not found: $SFT_CHECKPOINT" >&2
  echo "       Train from base model only, or set SFT_CHECKPOINT to a valid adapter dir." >&2
  exit 1
fi

if [[ -n "$SFT_ADAPTER_PATH" ]]; then
  if [[ ! -f "$SFT_ADAPTER_PATH/adapter_config.json" ]]; then
    echo "ERROR: SFT_ADAPTER_PATH missing adapter_config.json: $SFT_ADAPTER_PATH" >&2
    exit 1
  fi
  if [[ "$LORA_RANK" -le 0 ]]; then
    echo "ERROR: set LORA_RANK>0 when using SFT_ADAPTER_PATH" >&2
    exit 1
  fi
fi

TRAIN_PARQUET="$PARQUET_DIR/train.parquet"
LITE_FLAG=()
EPOCHS_DEFAULT=20
if [[ "$STAGE" == "rl1" ]]; then
  TRAIN_PARQUET="$PARQUET_DIR/train_lite.parquet"
  LITE_FLAG=(--lite_only)
  EPOCHS_DEFAULT=5
fi
MAX_TRAIN_SEQ_LEN="${MAX_TRAIN_SEQ_LEN:-18432}"
if [[ "$MAX_TRAIN_SEQ_LEN" -le 0 ]]; then
  echo "ERROR: MAX_TRAIN_SEQ_LEN must be > 0 for Gemma4-12B policy backward (got $MAX_TRAIN_SEQ_LEN)." >&2
  echo "       Unbounded SWE trajectories OOM ~80GB GPUs. Try 18432 (default), 16384, or 14336 if OOM." >&2
  exit 1
fi
EPOCHS="${EPOCHS:-$EPOCHS_DEFAULT}"

if [[ ! -f "$TRAIN_PARQUET" ]]; then
  echo "==> Building SkyRL parquet: $TRAIN_PARQUET"
  python integrations/skyrl_miniswe/preprocess_swegym.py \
    --output_dir "$PARQUET_DIR" \
    "${LITE_FLAG[@]}"
fi

OUT="${OUTPUT_DIR}-${STAGE}"
CKPT_PATH="${OUT}/checkpoints"
TRAJ_DIR="${OUT}/trajectories"
mkdir -p "$CKPT_PATH" "$TRAJ_DIR"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="skyrl-gemma4-12b-${STAGE}"
fi

EXTRA_ARGS=()
if [[ "$LORA_RANK" -gt 0 ]]; then
  EXTRA_ARGS+=(
    "trainer.policy.model.lora.rank=$LORA_RANK"
    "trainer.policy.model.lora.alpha=$LORA_ALPHA"
  )
  if [[ -n "$SFT_ADAPTER_PATH" ]]; then
    EXTRA_ARGS+=("trainer.policy.model.lora.adapter_path=$SFT_ADAPTER_PATH")
  fi
fi

# mini-swe-agent uses OpenAI tool_choice=auto; Gemma4 requires vLLM tool-call parser.
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-true}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-gemma4}"
EXTRA_ARGS+=(
  "generator.inference_engine.engine_init_kwargs.enable_auto_tool_choice=$VLLM_ENABLE_AUTO_TOOL_CHOICE"
  "generator.inference_engine.engine_init_kwargs.tool_call_parser=$VLLM_TOOL_CALL_PARSER"
)

echo "==> SkyRL GRPO stage=$STAGE policy=$POLICY_MODEL_PATH train=$TRAIN_PARQUET"
if [[ "$LORA_RANK" -gt 0 ]]; then
  echo "    policy LoRA: rank=$LORA_RANK alpha=$LORA_ALPHA adapter=${SFT_ADAPTER_PATH:-<random init>}"
  echo "    (ref LoRA in SkyRL logs may show rank=0; ref inherits policy adapter at runtime)"
fi
if [[ -n "${SFT_ADAPTER_PATH:-}" ]]; then
  echo "    SFT adapter init: $SFT_ADAPTER_PATH"
fi
echo "    placement: ${POLICY_GPUS} policy + ${NUM_ENGINES} vLLM engines (TP=$TP_SIZE, executor=$DISTRIBUTED_EXECUTOR_BACKEND, vllm_mem=$VLLM_GPU_MEMORY_UTILIZATION)"
echo "    weight sync: NCCL (trainer -> vLLM after each update)"
echo "    Docker rollouts: mode=$SKYRL_ROLLOUT_MODE queue=${SKYRL_ROLLOUT_QUEUE_HOST}:${SKYRL_ROLLOUT_QUEUE_PORT}"
echo "    vLLM HTTP for CPU: http://${HTTP_HOST}:${HTTP_PORT}/v1 (_SKYRL_USE_NEW_INFERENCE=$SKYRL_USE_NEW_INFERENCE)"
echo "    vLLM tools: enable_auto_tool_choice=$VLLM_ENABLE_AUTO_TOOL_CHOICE parser=$VLLM_TOOL_CALL_PARSER"
echo "    swanlab: mode=$SWANLAB_MODE logdir=$SWANLAB_LOG_DIR project=$PROJECT_NAME run=$RUN_NAME"
echo "    train seq cap: max_train_seq_len=$MAX_TRAIN_SEQ_LEN (text-only Gemma4; must be > 0)"

python -m integrations.skyrl_miniswe.main \
  "data.train_data=['$TRAIN_PARQUET']" \
  "data.val_data=['$PARQUET_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  "trainer.policy.model.path=$POLICY_MODEL_PATH" \
  trainer.placement.colocate_all=false \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node="$POLICY_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$POLICY_GPUS" \
  trainer.placement.policy_num_nodes=1 \
  trainer.placement.ref_num_nodes=1 \
  trainer.epochs="$EPOCHS" \
  trainer.eval_batch_size=16 \
  trainer.eval_before_train=false \
  trainer.eval_interval=5 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=6 \
  trainer.policy_mini_batch_size=3 \
  trainer.max_tokens_per_microbatch=18432 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.dump_data_batch=true \
  trainer.ckpt_interval=5 \
  trainer.max_prompt_length=4096 \
  trainer.algorithm.max_seq_len="$MAX_TRAIN_SEQ_LEN" \
  trainer.flash_attn=false \
  trainer.remove_microbatch_padding=false \
  generator.sampling_params.max_generate_length=4096 \
  generator.max_input_length=28672 \
  generator.max_train_seq_len="$MAX_TRAIN_SEQ_LEN" \
  generator.max_turns=20 \
  trainer.policy.optimizer_config.lr=5.0e-7 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.005 \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.num_engines="$NUM_ENGINES" \
  generator.inference_engine.tensor_parallel_size="$TP_SIZE" \
  generator.inference_engine.distributed_executor_backend="$DISTRIBUTED_EXECUTOR_BACKEND" \
  generator.inference_engine.enable_http_endpoint=true \
  "generator.inference_engine.http_endpoint_host=$HTTP_HOST" \
  generator.inference_engine.http_endpoint_port="$HTTP_PORT" \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.inference_engine.async_engine=true \
  generator.batched=true \
  generator.n_samples_per_prompt=4 \
  generator.inference_engine.gpu_memory_utilization="$VLLM_GPU_MEMORY_UTILIZATION" \
  "trainer.logger=$LOGGER" \
  "trainer.project_name=$PROJECT_NAME" \
  "trainer.run_name=$RUN_NAME" \
  trainer.resume_mode=null \
  "trainer.ckpt_path=$CKPT_PATH" \
  "generator.miniswe_config_path=$MINISWE_CONFIG" \
  "generator.miniswe_traj_dir=$TRAJ_DIR" \
  "${EXTRA_ARGS[@]}" \
  "$@"
