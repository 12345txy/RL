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

# 3. CPU (cpu-mechine-1): pull workers — 本机需同时 ssh work.bj11 + ssh cpu-mechine-1
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

## SSH 端口转发（work.bj11 + cpu-mechine-1）

| 端口 | 用途 | GPU `work.bj11` | CPU `cpu-mechine-1` |
|------|------|-----------------|---------------------|
| 6379 | Ray GCS | LocalForward | RemoteForward |
| 8265 | Ray Dashboard | LocalForward | — |
| 8001 | vLLM HTTP | LocalForward | RemoteForward |
| 9000 | Rollout queue | LocalForward | RemoteForward |

本机需**同时**连着 `ssh work.bj11` 与 `ssh cpu-mechine-1`，CPU 上访问 `127.0.0.1:8001` / `:9000` 经 RemoteForward 回到本机，再经 LocalForward 到 GPU。

## SFT checkpoint-150 → SWE-bench Verified（30 题）

**勿与 GRPO 训练同时占用 GPU :8001。**

```bash
# GPU (work.bj11)
LORA_PATH=outputs/sft-gemma4-12b-miniswe-lora/checkpoint-150 \
  PORT=8001 bash scripts/serve_gemma4_12b.sh

# CPU (cpu-mechine-1)，两条 SSH 隧道在线后
curl -s http://127.0.0.1:8001/v1/models | head
conda activate swebench
bash scripts/run_swebench_sft_ckpt150_eval.sh
```

Split：`data/splits/verified_eval30_seed42.json`（30 题，seed=42）。

## Notes

- SkyRL uses its **own** vLLM engines — do **not** run `serve_gemma4_12b.sh` during training.
- For standalone SWE-bench eval, serve SFT/RL LoRA on **:8001** (same port as pull rollout workers).
- Gemma4 may need vLLM version alignment; prefer **full SFT checkpoint** as `SFT_CHECKPOINT`.
