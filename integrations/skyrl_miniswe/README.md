# SkyRL + mini-swe-agent (true GRPO RL)

Real online RL with **vLLM weight sync** (NCCL) and **Docker rollouts on CPU Ray workers**.

## Architecture

```
GPU machine (8×H100)
├── 6 GPU: FSDP policy + ref (GRPO update)
├── 2 GPU: vLLM rollout engines (SkyRL-managed, weight sync via NCCL)
└── Ray head

CPU machine (Docker)
└── Ray worker (docker_node=1) → Mini-SWE-Agent Docker sandboxes
         └── HTTP → GPU vLLM :8001 (SkyRL internal endpoint)
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
# 1. GPU: Ray head (tunnel mode keeps GCS at 127.0.0.1 for SSH-forwarded workers)
bash scripts/run_skyrl_ray_head.sh

# 2. CPU: join cluster (swebench env; patches Ray client; SSH tunnels must stay open)
CONDA_ENV=swebench RAY_ADDRESS=127.0.0.1:6379 bash scripts/run_skyrl_ray_worker.sh
# Or manually on CPU after: python scripts/patch_ray_tunnel.py && export RAY_PRESERVE_LOCALHOST_IP=1

# 3. GPU: GRPO training (rl1 = lite pool, rl2 = full pool)
SFT_CHECKPOINT=outputs/sft-gemma4-12b-miniswe-full \
  SKYRL_HTTP_HOST=<GPU_IP> \
  SKYRL_REQUIRE_DOCKER_NODE=1 \
  STAGE=rl1 \
  bash scripts/run_rl_skyrl.sh
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
