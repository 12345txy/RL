# 本地改动记录

本文档记录本仓库相对 **上游默认行为** 做了哪些修改、为什么改、在新环境如何复现。**不是**「把旧机器文件 rsync 过去」的迁移清单，而是 **改动说明**。

改动分三类：

| 类型 | 位置 | 新环境怎么复现 |
|------|------|----------------|
| **A. vendor 补丁** | `vendor/SkyRL/` | 保持仓库内 diff，执行 `bash scripts/setup_skyrl.sh` |
| **B. 项目自研代码** | `integrations/`、`training/`、`data/` | 随 git 代码即可 |
| **C. 运行时补丁** | conda env 内的 Ray | 每次新建 `RL` 环境后自动/手动跑 `patch_ray_tunnel.py` |

SkyRL 补丁摘要另见：`vendor/SkyRL/PATCHES-SFT-ADAPTER.md`（本文更全）。

---

## 1. `vendor/SkyRL/` —— 相对上游 SkyRL 的补丁

上游 commit 基线（vendor 子目录内）：`ba94c5d`（2026-06 左右）。共 **23 个文件** 有 diff。

### 1.1 SFT LoRA → RL 冷启动（核心功能补丁）

**问题：** 上游 SkyRL 的 RL 只支持随机初始化 LoRA，无法从 SFT 的 PEFT checkpoint（如 `checkpoint-150`）继续训。

**改法：**

| 文件 | 改动 |
|------|------|
| `skyrl/train/config/config.py` | `SkyRLLoraConfig` 增加 `adapter_path: Optional[str]`；`TrainerConfig` 校验 `rank>0` 且目录含 `adapter_config.json` |
| `skyrl/backends/skyrl_train/workers/model_wrapper.py` | `HFModelWrapper` 增加参数 `lora_adapter_path`；若目录有效则 `PeftModel.from_pretrained(..., is_trainable=True)`，否则走原 `get_peft_model` |
| `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py` | Policy worker 传入 `lora_adapter_path`；**Ref worker 继承 policy 的 LoRA rank/adapter**（KL 参考同一 adapter） |

**使用：**

```bash
SFT_CHECKPOINT=outputs/sft-gemma4-12b-rebench-native-lora/checkpoint-20 \
STAGE=rl1 bash scripts/run_rl_skyrl.sh
# 等价于 trainer.policy.model.lora.adapter_path=...
```

---

### 1.2 Python 3.10 + 共享 RL conda 环境兼容

**问题：** 上游要求 Python 3.11，且 `pyproject.toml` 的 `fsdp` extra 强 pin vllm/torch/flash-attn，与项目已有的 **vLLM nightly + torch 2.11** 冲突。

**改法：**

| 文件 | 改动 |
|------|------|
| `pyproject.toml` | `requires-python = ">=3.10"`；`transformers` 上限改为 `<6.0.0`；**删除** fsdp extra 里对 `vllm==0.20.2`、`torch==2.11.0`、`flash-attn`、`flashinfer-*` 的 pin，改用 conda env 已有版本 |
| `skyrl/typing_compat.py` 等 | TypedDict / StrEnum 等 3.10 兼容（见 PATCHES 表） |
| Ray 相关 import | `PlacementGroupSchedulingStrategy` 改从 `ray.util.scheduling_strategies` 导入（Ray 2.55+） |

**安装方式：** `bash scripts/setup_skyrl.sh` 用 `pip install -e vendor/SkyRL --no-deps`，再装 `requirements-skyrl-rl.txt`，**不要** `pip install skyrl` 覆盖。

---

### 1.3 Gemma4 + 长 SWE 序列（显存 / 注意力）

**问题：** Gemma4-12B Unified 用 SDPA 而非 flash-attn；SWE 轨迹极长，RL 算 logprob 时 bf16 `log_softmax` 会 OOM。

**改法：**

| 文件 | 改动 |
|------|------|
| `skyrl/backends/skyrl_train/workers/model_wrapper.py` | `flash_attn.bert_padding` 改为 **try/except ImportError**；无 flash-attn 时仅在 `remove_microbatch_padding=true` 报错；Gemma4 默认 sdpa |
| `skyrl/backends/skyrl_train/utils/torch_utils.py` | `logprobs_from_logits_v2` 的 bf16 分支改为 **逐 token logsumexp**，避免 materialize `[seqlen, vocab]` |
| `skyrl/backends/skyrl_train/workers/model_wrapper.py` | logprob 路径对 `num_actions` 为 list 等情况的分支调整（配合 packing / 长序列） |

---

### 1.4 vLLM 0.22 权重同步（NCCL）

**问题：** vLLM nightly 已在 `GPUWorker` 内置 `start_weight_update` / `finish_weight_update`；SkyRL 自定义 wrap 与之重复，导致 GRPO 更新后 sync 失败。

**改法：**

| 文件 | 改动 |
|------|------|
| `skyrl/backends/skyrl_train/inference_servers/new_inference_worker_wrap.py` | 检测 vLLM 是否已有 native weight sync；有则 **delegate 给 vLLM 内置实现**，避免双份 wrap |
| `skyrl/backends/skyrl_train/inference_engines/vllm/vllm_engine.py` 等 | 小改动配合 vLLM 0.22 接口 |

**相关脚本默认：** `run_rl_skyrl.sh` 里 `TP_SIZE=1` 时 `DISTRIBUTED_EXECUTOR_BACKEND=mp`（避免 vLLM `executor=ray` 在 SSH 隧道下二次 `ray.init()` 卡死）。

---

### 1.5 PEFT / LoRA 同步小修复

| 文件 | 改动 |
|------|------|
| `skyrl/backends/skyrl_train/workers/fsdp/fsdp_worker.py` | 导出 LoRA 配置时 `task_type` / `peft_type` 可能是 enum 或已是 str，**兼容两种** |

---

### 1.6 如何在新 GPU 环境复现 SkyRL 补丁

```bash
cd ~/RL
git clone / checkout 含 vendor diff 的分支
grep -q adapter_path vendor/SkyRL/skyrl/train/config/config.py || echo "补丁缺失"
bash scripts/setup_skyrl.sh
```

若 vendor 目录被误覆盖，在 `vendor/SkyRL` 内对上游 `ba94c5d`（或你记录的 base commit）重新 `git apply` 本仓库保存的 patch。

---

## 2. SFT 训练 —— 项目自研（非 vendor 补丁）

### 2.1 DeepSpeed ZeRO-3 + 长上下文 chunked loss

**文件：** `training/chunked_nll_deepspeed.py`

**问题：** TRL `chunked_nll` 在 ZeRO-3 下 `lm_head` 分片，直接算 CE 会报错或 OOM。

**改法：** 运行时 monkey-patch `trl.trainer.sft_trainer._chunked_cross_entropy_loss`：每次 loss 前 **all_gather 完整 lm_head**，再分 chunk 算 CE。

**触发：** `scripts/train_sft.py` 在 `loss_type=chunked_nll` 且 `deepspeed_zero_stage=3` 且多卡时调用 `patch_chunked_nll_for_deepspeed_zero3()`。

---

### 2.2 Gemma4 LoRA target modules

**文件：** `training/gemma_lora.py`

**问题：** Gemma4 Unified 语言层路径为 `model.language_model.layers.{i}.self_attn.*`，与通用 `q_proj` 短名不匹配。

**改法：** 检测模型名含 `gemma` 时，按层展开完整 module 路径作为 `target_modules`。

---

### 2.3 SFT 数据管线（native tool）

**文件：** `data/swe_utils.py`、`data/prepare_rebench_native_sft.py`

**改动要点（相对旧版 mini-swe THOUGHT/ACTION）：**

- `looks_like_native_tool_format()` / `normalize_tool_calls()`：校验 OpenAI native tool 格式
- `prepare_gemma4_chat_messages()`：Gemma4 chat template + `tool_call_id` 配对
- ReBench 225 条 resolved 轨迹 → `data/sft/sft_merged.jsonl`
- **无验证集**：`train_sft.py` 仅 `train_dataset`，无 `eval_dataset`（过拟合需训后 agent 评测）

---

## 3. SkyRL + mini-swe-agent 集成 —— 项目自研

**目录：** `integrations/skyrl_miniswe/`

| 文件 | 作用 |
|------|------|
| `main.py` | SkyRL GRPO 入口 |
| `generator.py` / `rollout_worker.py` | CPU Docker rollout |
| `mini_swe_utils.py` | patch 评测、`eval_script` reward |
| `rollout_core.py` | pull 模式队列消费 |
| `swebench.yaml` | agent prompt（与 ReBench 数据里的 submit 命令应对齐） |
| `preprocess_swegym.py` | SWE-Gym → parquet |

**相对上游 SkyRL 无 patch**，是本仓库新增集成层；新环境随代码部署即可。

---

## 4. 运行时补丁（改 conda 环境，不在 git 里）

### 4.1 Ray SSH 隧道 —— `scripts/patch_ray_tunnel.py`

**问题：** Ray 会把 `--node-ip-address=127.0.0.1` 解析成容器内网 IP，CPU worker 经 SSH 隧道连 `127.0.0.1:6379` 会失败。

**改法：** 修改 **当前 conda env** 内 `ray/_private/services.py` 的 `resolve_ip_for_localhost`：当 `RAY_PRESERVE_LOCALHOST_IP=1` 时直接返回 `127.0.0.1`。

**复现：**

```bash
conda activate RL
export RAY_PRESERVE_LOCALHOST_IP=1
python scripts/patch_ray_tunnel.py
```

`run_rl_skyrl.sh`、`run_skyrl_ray_head.sh` 在 `RAY_TUNNEL_MODE=1` 时会自动调用。**重建 conda 环境后必须再跑一次。**

---

### 4.2 Ray Head 启动 hook —— `integrations/skyrl_miniswe/ray_tunnel_hook.py`

**问题：** 同上，head 广播地址需保持 loopback。

**改法：** `RAY_START_HOOK=integrations.skyrl_miniswe.ray_tunnel_hook.patch_ray_params_for_ssh_tunnel`（`run_skyrl_ray_head.sh` 默认设置），强制 head 的 `node_ip_address=127.0.0.1`。

---

## 5. 评测 / 推理 —— 行为配置（非上游 patch）

### 5.1 vLLM sticky routing —— `integrations/miniswe/sticky_routing.py`

**问题：** 8 实例 vLLM + prefix cache 需要同一 SWE instance 固定到同一 GPU。

**改法：** monkey-patch `minisweagent.run.benchmarks.swebench.process_instance`，给每次 LLM 请求加 header `X-SWE-Instance-Id`。

**配合：** `scripts/nginx_vllm_lb.sh` 默认 `NGINX_LB_MODE=sticky`，按该 header hash 到 backend。

---

### 5.2 脚本层默认值（按环境调整，不是代码 patch）

| 脚本 | 常见需改项 |
|------|-----------|
| 多数 `scripts/*.sh` | 硬编码 `source /root/miniconda3/...` → 可改为 `$(conda info --base)` |
| `scripts/run_rl_skyrl.sh` | `POLICY_GPUS=6`、`ROLLOUT_GPUS=2`；`SKYRL_HTTP_HOST`；`SFT_CHECKPOINT` |
| `scripts/run_sft.sh` | `ACCEL_CONFIG=configs/accelerate_deepspeed_zero3_8gpu.yaml`（GPU 数） |
| `configs/swebench_docker_gemma4_12b.yaml` | `model.model_kwargs.api_base`（应用 `VLLM_BASE` 环境变量覆盖） |

---

## 6. 新环境最小复现 checklist

```bash
# 1. 代码（含 vendor/SkyRL diff）
git checkout <branch>

# 2. Conda + SkyRL editable
conda create -y -n RL python=3.10
conda activate RL
bash scripts/setup_skyrl.sh

# 3. Ray 隧道补丁（若 CPU↔GPU 走 SSH）
export RAY_PRESERVE_LOCALHOST_IP=1 RAY_TUNNEL_MODE=1
python scripts/patch_ray_tunnel.py

# 4. 验证 SkyRL 核心补丁
grep adapter_path vendor/SkyRL/skyrl/train/config/config.py
python -c "from vllm.config import WeightTransferConfig; print('ok')"

# 5. SFT / vLLM 依赖
pip install -r requirements.txt   # + vLLM nightly for Gemma4
```

---

## 7. 改动与文档索引

| 文档 / 文件 | 内容 |
|-------------|------|
| 本文 | 全部本地改动汇总 |
| `vendor/SkyRL/PATCHES-SFT-ADAPTER.md` | SkyRL 补丁简表 |
| `integrations/skyrl_miniswe/README.md` | RL 架构 + SSH 端口 |
| `docs/swe_rl_pipeline.md` | 阶段流程 |
| `docs/weekly_report_sft_rl.md` | SFT/RL 方案背景 |

---

**维护建议：** 每次改 `vendor/SkyRL`，在 vendor 目录内 `git diff > ../../patches/skyrl-local.patch` 备份，并更新本节表格；升级上游 SkyRL 时按 patch 冲突逐项 merge。
