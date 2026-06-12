# Gemma4-12B SWE-RL Pipeline (mini-swe-agent + SWE-Gym + 8×H100)

## Quick start

```bash
# 0. Environments
bash scripts/setup_swebench_vm.sh   # mini-swe-agent eval (CPU VM + Docker)
conda activate RL                   # SFT/RL training
conda activate swebench             # SWE-bench eval

# 1. Download model
bash scripts/download_gemma4_12b.sh

# 2. Phase 0 — baseline
bash scripts/serve_gemma4_12b.sh   # GPU terminal
bash scripts/run_miniswe_baseline.sh

# 3. Phase 1 — data
bash scripts/run_prepare_data.sh

# 4. Phase 2 — SFT
STAGE=lora bash scripts/run_sft.sh
STAGE=full bash scripts/run_sft.sh

# 5. Phase 3 — Agent RL
CHECKPOINT=outputs/sft-gemma4-12b-miniswe-full bash scripts/serve_gemma4_12b.sh
STAGE=rl1 bash scripts/run_rl_skyrl.sh
STAGE=rl2 bash scripts/run_rl_skyrl.sh

# 6. Phase 4 — Verifier + Verified
bash scripts/run_verifier.sh

# SWE-bench eval (any checkpoint)
VLLM_BASE=http://127.0.0.1:8000/v1 SLICE=0:100 WORKERS=2 \
  bash scripts/run_swebench_vm_docker.sh
```

## Data layout

| Path | Description |
|------|-------------|
| `data/sft/miniswe_train.jsonl` | mini-swe-agent SFT trajectories |
| `data/sft/sft_merged.jsonl` | Merged SFT |
| `data/rl/swegym_rl_train.jsonl` | RL pool (Verified leakage removed) |
| `data/splits/verified_dev_100.json` | Dev holdout split |

## GPU layout (RL)

- 6 GPUs: policy training (FSDP via accelerate)
- 2 GPUs: vLLM async rollout (`NUM_INSTANCES=2`)
- CPU VM: mini-swe-agent Docker environments

## Sky-RL

Full Sky-RL integration config: `integrations/skyrl_miniswe/config.yaml`

Standalone rollout + GRPO-style buffer: `scripts/train_agent_rl.py`
