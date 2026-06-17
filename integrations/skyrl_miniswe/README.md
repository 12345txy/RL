# SkyRL + mini-swe-agent (true GRPO RL)

Real online RL with **vLLM weight sync** (NCCL) and **Docker rollouts on CPU Ray workers**.

## Architecture

```
GPU machine (8×H100)
├── 6 GPU: FSDP policy + ref (GRPO update)
├── 2 GPU: vLLM rollout engines (SkyRL-managed, weight sync via NCCL)
└── Ray head

CPU machine (Docker)
└── Pull rollout workers → Mini-SWE-Agent Docker sandboxes
         ├── HTTP pull  GPU rollout queue :9000
         └── HTTP call  GPU vLLM :8001 (SkyRL internal endpoint)
```

After each GRPO step, SkyRL pushes updated weights to vLLM — **no manual restart**.

## Setup

```bash
# GPU machine
bash scripts/setup_skyrl.sh
conda activate RL

# CPU machine (Docker VM)
bash scripts/setup_skyrl.sh   # or: pip install ray mini-swe-agent
bash scripts/setup_swebench_vm.sh
```

## Run

```bash
# 1. GPU: Ray head
bash scripts/run_skyrl_ray_head.sh

# 2. GPU: GRPO training (starts rollout queue on :9000 by default)
SFT_CHECKPOINT=outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150 \
  SKYRL_HTTP_HOST=127.0.0.1 \
  STAGE=rl1 \
  bash scripts/run_rl_skyrl.sh

# 3. CPU: pull workers (no Ray worker / docker_node needed)
#    SSH tunnels must forward 9000 (queue) and 8001 (vLLM) like existing 6379/8001
CONDA_ENV=swebench bash scripts/run_rollout_pull_worker.sh
```

## Files

| File | Role |
|------|------|
| `main.py` | SkyRL `BasePPOExp` entrypoint |
| `generator.py` | `MiniSweAgentGenerator` (real Docker + harness reward) |
| `mini_swe_utils.py` | Docker env + patch evaluation |
| `swebench.yaml` | Mini-SWE-Agent prompt + Docker config |
| `preprocess_swegym.py` | SWE-Gym → parquet for SkyRL |
| `config.yaml` | Reference hyperparameters |

## Patched SkyRL in shared ``RL`` env

Uses existing **vLLM nightly + torch 2.11** (no downgrade). Install:

```bash
bash scripts/setup_skyrl.sh   # conda activate RL
```

Patches: `vendor/SkyRL/` — SFT adapter init, relaxed deps, Python 3.10+.

从 SFT adapter 继续 RL（例如 `checkpoint-150`）：

```bash
SFT_CHECKPOINT=outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150 \
  POLICY_MODEL_PATH=models/gemma-4-12B-it \
  STAGE=rl1 \
  bash scripts/run_rl_skyrl.sh
```

等价于设置：

```bash
trainer.policy.model.path=models/gemma-4-12B-it
trainer.policy.model.lora.rank=64
trainer.policy.model.lora.adapter_path=outputs/.../checkpoint-150
```

SkyRL patch 位置：`vendor/SkyRL/`（`adapter_path` 加载 + ref 同步同一 adapter）。

## Legacy

The old pseudo-RL loop (`scripts/train_agent_rl.py` + manual vLLM restart) is deprecated.
Use `bash scripts/run_rl_legacy.sh` only for debugging.

## Notes

- SkyRL uses its **own** vLLM engines — do **not** run `serve_gemma4_12b.sh` during training.
- For SWE-bench **eval** after RL, export checkpoint and serve separately on GPU.
- Gemma4 may need vLLM version alignment; prefer **full SFT checkpoint** as `SFT_CHECKPOINT`.
