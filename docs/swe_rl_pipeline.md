# Gemma4-12B SWE-RL Pipeline (SkyRL + mini-swe-agent + SWE-Gym + 8×H100)

## Quick start

```bash
# 0. Environments
bash scripts/setup_swebench_vm.sh   # CPU VM: Docker + mini-swe-agent
bash scripts/setup_skyrl.sh         # GPU: patched SkyRL into shared RL env (vLLM 0.22 nightly)
conda activate RL                   # SFT + RL share this env (Gemma4 needs vLLM nightly)
conda activate swebench             # SWE-bench eval only

# 1. Download model
bash scripts/download_gemma4_12b.sh

# 2. Phase 0 — baseline
bash scripts/serve_gemma4_12b.sh   # GPU terminal (eval only)
bash scripts/run_miniswe_baseline.sh

# 3. Phase 1 — data
bash scripts/run_prepare_data.sh

# 4. Phase 2 — SFT
STAGE=lora bash scripts/run_sft.sh
STAGE=full bash scripts/run_sft.sh

# 5. Phase 3 — True Agent RL (SkyRL GRPO + weight sync)
# GPU machine:
bash scripts/run_skyrl_ray_head.sh
# CPU machine (Docker):
RAY_ADDRESS=<GPU_IP>:6379 bash scripts/run_skyrl_ray_worker.sh
# GPU machine:
SFT_CHECKPOINT=outputs/sft-gemma4-12b-miniswe-full \
  SKYRL_HTTP_HOST=<GPU_IP> \
  SKYRL_REQUIRE_DOCKER_NODE=1 \
  STAGE=rl1 bash scripts/run_rl_skyrl.sh
STAGE=rl2 bash scripts/run_rl_skyrl.sh

# 6. Phase 4 — Verifier + Verified
bash scripts/run_verifier.sh

# SWE-bench eval (export RL checkpoint, then serve on GPU)
CHECKPOINT=outputs/rl-gemma4-12b-miniswe-rl2/checkpoints/... \
  bash scripts/serve_gemma4_12b.sh
VLLM_BASE=http://127.0.0.1:8000/v1 SLICE=0:100 WORKERS=2 \
  bash scripts/run_swebench_vm_docker.sh
```

## Phase 3: True RL (SkyRL)

Unlike the legacy `train_agent_rl.py` loop, SkyRL:

1. Runs **GRPO policy gradient** on 6 GPUs (FSDP)
2. Runs **vLLM rollout** on 2 GPUs (SkyRL-managed)
3. **Syncs weights to vLLM via NCCL** after each update (no manual restart)
4. Schedules **Docker rollouts on CPU Ray workers** (`docker_node` resource)

Do **not** run `serve_gemma4_12b.sh` during SkyRL training.

Legacy pseudo-RL: `bash scripts/run_rl_legacy.sh`

## Data layout

| Path | Description |
|------|-------------|
| `data/sft/rebench_native_train.jsonl` | SWE-ReBench native-tool SFT (primary) |
| `data/sft/sft_merged.jsonl` | Merged SFT (default train entry) |
| `data/rl/swegym_rl_train.jsonl` | Legacy RL pool (jsonl) |
| `data/rl/skyrl_parquet/train.parquet` | SkyRL full train set |
| `data/rl/skyrl_parquet/train_lite.parquet` | SkyRL lite (rl1) |
| `data/rl/skyrl_parquet/validation.parquet` | SkyRL validation |
| `data/splits/verified_dev_100.json` | Dev holdout split |

## GPU / CPU layout

| Machine | Role |
|---------|------|
| **GPU (8×H100)** | Ray head, FSDP policy (6 GPU), vLLM engines (2 GPU), NCCL weight sync |
| **CPU VM** | Ray worker, Docker sandboxes for Mini-SWE-Agent rollouts |

## Config reference

`integrations/skyrl_miniswe/config.yaml`
