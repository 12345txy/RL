# 周报：Gemma4 SWE-RL 流水线跑通与评测诊断

**周期：** 2026.06.16 – 2026.06.23

---

## 核心结论（TL;DR）

| 维度 | 结论 |
|------|------|
| **工程** | SkyRL 真在线 GRPO 流水线已端到端跑通（训练 ↔ 推理权重同步 ↔ Docker rollout） |
| **RL 效果** | 当前所有 rollout **reward = 0**，尚无 resolved 样本，策略几乎收不到正反馈 |
| **SFT 效果** | SWE-bench Verified 30 题：**SFT 0 提交**，base 略好但仍极差；SFT 未带来可测提升 |
| **根因判断** | SFT 数据 tool_call 占比低、与 RL 推理路径不一致，可能训偏；12B 本身长程 SWE 能力弱 |
| **下一步** | **计划直接在 Gemma4-31B base 上做 RL的尝试** |

---

## 一、本周完成事项

1. **联调跑通** Gemma4 + SkyRL GRPO + mini-swe-agent 完整 RL 流水线（端到端训练闭环）
2. **发现问题** RL的reward恒为0，使用sft的ckpt测试了swebench-v随机抽样的30道题没有一道题成功提交patch,case study发现主要问题是没法稳定调用tool_call,疑似sft数据中的tool_call内容占比太低，测试效果不如base_model(虽然base的表现也很差)
3. 尝试启动 **Gemma4-31B base 全参 RL** 实验，作为后续主线

---

## 二、RL 技术路线

### 2.1 系统架构

```
GPU（8×H100）                         CPU（Docker）
┌─────────────────────────┐          ┌──────────────────────────┐
│ FSDP Policy  ── GRPO 更新│          │ Pull worker 拉取任务      │
│ FSDP Ref     ── KL 参考  │  SSH隧道  │ mini-swe-agent 多轮交互   │
│ vLLM         ── 采样推理 │ ◄──────► │ Docker 沙箱执行 bash/patch│
│ NCCL 权重同步 ◄──────────│          │ 调 GPU vLLM 生成下一步     │
└─────────────────────────┘          └──────────────────────────┘
```

**GPU 侧职责（6 + 2 卡）：**

- **6 卡**：FSDP 训练 Policy（待优化策略）与 Ref（KL 锚点），colocate 在同一组 GPU 上交替使用
- **2 卡**：vLLM rollout 引擎，对外提供 OpenAI 兼容 HTTP 接口供 CPU agent 调用
- **Ray**：调度训练 worker；Rollout 默认走 **Pull 模式**（CPU 主动拉任务），不强制 CPU 加入 Ray 集群

**CPU 侧职责：**

- 从 GPU 上的任务队列取 SWE 实例
- 在 Docker 里跑完整 agent 循环：读 issue → 调模型 → 执行命令 → 编辑代码 → 提交 patch
- 把轨迹和 **0/1 reward** 回传给 GPU 参与 GRPO

### 2.2 算法：GRPO + 稀疏二值 Reward

**GRPO（Group Relative Policy Optimization）** 是本项目的 RL 算法：

- 每个 prompt（一个 SWE 实例）采样 **多条轨迹**（默认 4 条）
- Reward 只在 episode 结束给出：**测试通过 → 1，否则 → 0**
- 在组内做相对 advantage：同 prompt 下「相对更好」的轨迹被强化，差的被抑制
- 配合 **KL loss**（系数 0.005）约束策略不要偏离 Ref 太远，稳定训练

**数据分阶段：**

- **rl1**：SWE-Gym-Lite 子集，小规模验证流水线
- **rl2**：全量 SWE-Gym，正式训练

---

## 三、实验发现与问题分析

### 3.1 RL 训练：Reward 全部为 0

流水线各阶段均已打通（rollout → 回传 → GRPO update → 权重同步），但 **尚未观测到任何 resolved 样本**，训练 reward 恒为 0。

**Agent 侧常见终止原因：**

- `LimitsExceeded`：250 步内未完成修复（一直没有tool_calling）
- `Timeout`：单轮生成过长（600s），占满 agent 预算
- 从未走到 submit，patch 为空

### 3.2 SWE-bench Verified 30 题在 Gemma4-12B-SFT 上的统计结果

30 题均无成功提交 patch（**0 / 30 Submitted**）。按退出状态分布如下：

| 退出状态 | 数量 | 说明 |
|----------|------|------|
| LimitsExceeded | 12 | 250 步耗尽；全部为「No tool calls found」 |
| TimeoutExpired | 13 | Docker 起容器 120s 超时；无轨迹 |
| InternalServerError | 3 | vLLM 断连 |
| Timeout | 2 | LiteLLM 600s 超时 |
| FormatError 类型 | 3342 次（17 题有轨迹） | 100% 为无 tool_call |

**结论：当前 SFT 没有带来可衡量的 SWE 能力提升，部分 case 还不如 base。**

### 3.3 Case Study：SFT 可能「训偏了」

对比 SFT 数据、RL rollout 路径与失败轨迹，主要发现：

**1. 数据格式与 RL 推理路径不一致**

- SFT 数据来自 mini-swe 专家轨迹，大量是 **纯文本 THOUGHT/ACTION** 格式
- RL rollout 使用 vLLM **native tool-calling**（结构化 tool_call token）
- 模型在 SFT 阶段学到的交互模式，与 RL 阶段实际使用的接口不对齐

**2. tool_call 占比偏低**

- SFT 集中教「怎么写长推理、怎么拼 bash 字符串」
- 对「稳定、多轮、结构化地调用工具」覆盖不足
- 表现为：爱长篇生成、不爱 submit、工具调用不稳定

**3. Gemma4-12B 能力天花板**

- 即便用 base，在 SWE-bench 上仍频繁 **超时、步数超限**
- 长上下文 + 多轮编辑 + 定位 bug，对 12B 来说任务过重
- 问题不全是 SFT 的锅，**基座规模本身也是瓶颈**

---

## 四、进展与下一步

### 当前状态

| 项 | 状态 |
|----|------|
| 真在线 GRPO 流水线 | ✅ |
| RL 非零 reward | ❌ |
| SFT 评测有效提升 | ❌ |
| 31B base RL | 🔄 进行中 |

### 下一步：Gemma4-31B 直接做 RL + 进一步分析评测相关的参数和trick

**战略调整：** 不再以 12B SFT 为 RL 起点，改为 **31B base 冷启动全参 GRPO**。

**理由：**

1. 12B SFT 在 SWE-bench 上已验证无效，且可能与 RL tool-calling 路径冲突
2. 31B 容量更大，长程推理与工具使用潜力更高
3. 全参 RL 避免跨模型 LoRA 适配问题

**资源与配置思路（相对 12B）：**

- 训练：6 卡 FSDP 全参 Policy/Ref
- 推理：2 卡 vLLM，Tensor Parallel = 2
- 序列上限适当下调（如 16k）以控制显存
- 仍用 SWE-Gym-Lite 做 rl1 冒烟，再扩到全量

**并行优化方向（与模型规模无关）：**

- 限制单轮生成长度，避免一步吃满 timeout
- 降低 rollout 并发，对齐 vLLM 吞吐
- 若 reward 持续稀疏，考虑课程学习或 reward shaping
- Agent 配置与 RL 推理路径统一为 tool-calling 范式

---


