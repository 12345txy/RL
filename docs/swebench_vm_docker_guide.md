# SWE-bench 配置指南：vLLM API + 虚拟机全栈

本指南采用 **固定分工**：

| 机器 | 职责 | 不需要 |
|------|------|--------|
| **GPU 机器**（Workshop 等） | 只跑 **vLLM**，提供 OpenAI 兼容 API | Docker、mini-swe-agent、swebench、Modal、sb-cli |
| **CPU 虚拟机** | **其余全部**：agent、Docker 沙箱、harness 评测 | GPU、vLLM |

```
                    OpenAI API (HTTP)
  ┌──────────────────────────┐         ┌─────────────────────────────┐
  │  GPU 机器                 │         │  CPU 虚拟机                  │
  │                          │  :8000  │                             │
  │  vLLM serve              │ ◄────── │  mini-swe-agent (Docker)    │
  │  Gemma4-12B              │         │  swebench.harness (Docker)  │
  │  scripts/serve_gemma4_12b│         │  preds.json / 轨迹 / 报告    │
  └──────────────────────────┘         └─────────────────────────────┘
         仅推理                              环境 + 跑题 + 评测
```

**不使用** Modal 沙箱、**不使用** sb-cli 云端评测。

---

## 0. 两台机器各自要做什么（一览）

### GPU 机器（3 步）

1. 有项目代码（至少 `scripts/serve_gemma4_12b.sh` 和模型权重）
2. 启动 vLLM：`bash scripts/serve_gemma4_12b.sh`
3. 让 VM 能访问 `8000` 端口（SSH 隧道或防火墙放行）

### CPU 虚拟机（全部 SWE-bench 流程）

1. 安装 Docker
2. 克隆项目、`bash scripts/setup_swebench_vm.sh`
3. SSH 隧道连 GPU 上的 vLLM
4. `bash scripts/run_swebench_vm_docker.sh` → 生成 `preds.json`
5. `bash scripts/eval_swebench_local.sh` → 本地 harness 打分

---

## 1. 虚拟机硬件建议

| 资源 | 建议 |
|------|------|
| CPU | 8 核+ |
| 内存 | **32GB+**（`WORKERS=2`）；并行更多建议 64GB |
| 磁盘 | **100GB+ 空闲**（SWE-bench Docker 镜像按 repo 拉取，累计可达数十 GB） |
| GPU | **不需要** |

---

## 2. GPU 机器：只启动 vLLM API

```bash
cd ~/working/RL   # 或你的项目路径
bash scripts/serve_gemma4_12b.sh
```

默认：`MAX_MODEL_LEN=131072`，端口 `8000`，模型 `gemma-4-12B-it`。

在 **GPU 机器本机** 确认服务正常：

```bash
curl -s http://127.0.0.1:8000/v1/models | head
```

GPU 机器上 **无需** 安装 Docker、mini-swe-agent、swebench，也 **不要** 跑 `run_swebench_gemma4_12b.sh`（那是 Modal 方案）。

---

## 3. 虚拟机：安装 Docker（Ubuntu 22.04/24.04）

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker "$USER"
newgrp docker   # 或重新登录 SSH

docker run --rm hello-world
docker ps       # 必须无报错
```

---

## 4. 虚拟机：同步代码与 Python 环境

```bash
git clone <你的仓库地址> RL
cd RL

bash scripts/setup_swebench_vm.sh
conda activate swebench
```

| 文件 | 作用 |
|------|------|
| `requirements-swebench-docker.txt` | VM 依赖（mini-swe-agent + swebench，无 Modal） |
| `configs/swebench_docker_gemma4_12b.yaml` | Docker 沙箱 + 远程 vLLM |
| `scripts/setup_swebench_vm.sh` | 安装 Python 环境 |
| `scripts/run_swebench_vm_docker.sh` | Agent 跑题 → `preds.json` |
| `scripts/eval_swebench_local.sh` | 本地 harness 评测 |

自定义 conda 环境名：

```bash
CONDA_ENV=myenv bash scripts/setup_swebench_vm.sh
```

---

## 5. 虚拟机 → GPU：连接 vLLM API

VM 上的 agent 通过 `VLLM_BASE` 调用 GPU 上的 vLLM（LiteLLM `hosted_vllm/*`）。

Workshop 上 vLLM 默认监听 **`0.0.0.0:8000`**。在 Workshop 内确认：

```bash
curl -s http://127.0.0.1:8000/v1/models | head
hostname -I   # 记下内网 IP，例如 172.19.x.x
```

VM 与 Workshop 往往 **不在同一二层网络**，SSH `-L`/`-R` 隧道可能不通（已实测失败时跳过方式 A）。

### 方式 A：VKS Server Export（平台推荐）

在 VKS 控制台对 GPU 实例：

1. 进入 **Run Shell**（或任意终端）启动 vLLM：`bash scripts/serve_gemma4_12b.sh`
2. 开启 **Server Export**，导出端口 **8000**
3. 复制平台给出的 **外网/内网访问地址**

在 VM 上验证（把 URL 换成平台给的）：

```bash
curl -s http://<export-地址>/v1/models | head
export VLLM_BASE=http://<export-地址>/v1
```

**不必依赖 Run Shell UI 才能起 vLLM**——Cursor/SSH 终端里启动即可；Export 只是在平台侧登记端口转发。若平台提供 **CLI/API** 做 Export，与 UI 等价，向 VKS 管理员索取命令（常见为 `kubectl port-forward` 或平台自有 `vks` 子命令）。

### 方式 B：内网 IP 直连（同一 VPC 时最快）

若 VM 能路由到 Workshop 内网 IP，在 **VM** 上：

```bash
curl -s http://172.19.37.205:8000/v1/models | head   # IP 换成 hostname -I 的结果
export VLLM_BASE=http://172.19.37.205:8000/v1
```

不通则说明两机网络隔离，改用方式 A 或 C。

### 方式 C：cloudflared 临时公网隧道（纯 CLI，不依赖 VKS Export）

在 **GPU Workshop** 安装并启动（与 Run Shell 无关，SSH 终端即可）：

```bash
# 一次性下载或使用已安装的 cloudflared
cloudflared tunnel --url http://127.0.0.1:8000
```

终端会打印类似 `https://xxxx.trycloudflare.com` 的 URL。在 VM 上：

```bash
curl -s https://xxxx.trycloudflare.com/v1/models | head
export VLLM_BASE=https://xxxx.trycloudflare.com/v1
```

注意：临时 URL 每次重启会变；仅适合调试，生产环境优先用方式 A。

### 方式 D：SSH 隧道（仅当 VM↔Workshop SSH 互通且 sshd 允许转发时）

Workshop → VM 反向：

```bash
ssh -N -R 8000:127.0.0.1:8000 user@VM_IP
```

VM → Workshop 正向：

```bash
ssh -N -L 8000:127.0.0.1:8000 user@WORKSHOP
```

VM 上 `curl http://127.0.0.1:8000/v1/models`。**若已证实不通，不要用此方式。**

---

## 6. 虚拟机：运行 Agent（生成 preds.json）

```bash
conda activate swebench
cd ~/RL

VLLM_BASE=http://127.0.0.1:8000/v1 \
SLICE=0:100 \
WORKERS=2 \
OUTPUT_DIR=results/swebench_vm_docker_100 \
bash scripts/run_swebench_vm_docker.sh
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `VLLM_BASE` | `http://127.0.0.1:8000/v1` | GPU 上 vLLM 的 API 地址 |
| `SLICE` | `0:5` | 题目范围，如 `0:100` |
| `WORKERS` | `2` | VM 上并行题数；内存小用 `1` |
| `REDO_EXISTING` | `0` | 设为 `1` 覆盖已有轨迹 |

输出目录：

- `results/.../preds.json` — 提交给 harness 的 patch
- `results/.../<instance_id>/` — 各题轨迹

首次运行会 `docker pull` `swebench/sweb.eval.*` 镜像，耗时较长。

---

## 7. 虚拟机：Harness 评测

Agent 完成后，**仍在同一台 VM** 上评测，无需 sb-cli：

```bash
conda activate swebench
cd ~/RL

bash scripts/eval_swebench_local.sh \
  results/swebench_vm_docker_100/preds.json \
  gemma4-vm-docker-100
```

| 变量 | 默认 | 说明 |
|------|------|------|
| `MAX_WORKERS` | `4` | harness 并行评测容器数 |
| `CLEAN` | `0` | 设为 `1` 强制重跑所有 instance |

日志：`results/swebench_local_reports/<run_id>.log`  
详细结果：`logs/run_evaluation/<run_id>/`（swebench 默认路径）

---

## 8. 推荐参数（Gemma4-12B）

| 场景 | VM `WORKERS` | `SLICE` | GPU `MAX_MODEL_LEN` |
|------|--------------|---------|---------------------|
| 调试 | 1 | `0:5` | 131072 |
| 100 题 | 2 | `0:100` | 131072 |
| VM 内存紧张 | 1 | — | 131072 |
| GPU 显存紧张 | — | — | 65536（易 context 爆，不推荐） |

GPU 侧 context 不够时在 **GPU 机器** 调整：

```bash
MAX_MODEL_LEN=131072 bash scripts/serve_gemma4_12b.sh
```

VM 侧 context 报错时，先把 `WORKERS` 降到 `1`，减轻 vLLM 并发压力。

---

## 9. 故障排查

### VM：`Cannot connect to the Docker daemon`

```bash
sudo systemctl status docker
groups | grep docker
```

### VM：`vLLM not reachable`

1. GPU 机器上 vLLM 是否在跑：`curl http://127.0.0.1:8000/v1/models`  
2. `VLLM_BASE` 是否指向 **Server Export / 内网 IP / cloudflared** 的有效地址（不要假设 SSH 隧道可用）  
3. 在 VM 上：`curl -v "$VLLM_BASE/models"` 看是连接超时还是 HTTP 4xx/5xx  
4. Export 地址是否含 `/v1` 后缀（应为 `.../v1`，不是只到 host）  

### Agent：`ContextWindowExceededError`

- 在 **GPU 机器** 增大 `MAX_MODEL_LEN`  
- 在 **VM** 设 `WORKERS=1`  

### Agent：`LimitsExceeded` / 步数打满

- 弱模型易空转；可在 `configs/swebench_docker_gemma4_12b.yaml` 调整 `agent.step_limit`（默认 200）

### Docker 磁盘占满

```bash
docker system df
docker image ls | grep swebench
```

### Harness：patch apply 失败

查看 `results/swebench_local_reports/<run_id>.log`，检查 `preds.json` 中 patch 是否完整。

---

## 10. 与旧方案（全在 Workshop + Modal）的区别

| 环节 | 旧方案（Workshop 无 Docker） | **本方案** |
|------|------------------------------|------------|
| LLM 推理 | Workshop GPU vLLM | **GPU 机器 vLLM API**（不变） |
| Agent 沙箱 | Modal 云端 | **VM Docker** |
| 评测 | sb-cli 云端 | **VM swebench.harness** |
| GPU 机器还要装 | Modal、sb-cli 等 | **仅 vLLM** |
| VM 还要装 | — | Docker + mini-swe-agent + swebench |

---

## 11. 快速命令清单

```bash
# ── GPU 机器（只做推理 API）──
cd ~/working/RL
bash scripts/serve_gemma4_12b.sh
curl -s http://127.0.0.1:8000/v1/models | head

# ── CPU 虚拟机：验证 vLLM（任选一种已打通的方式）──
curl -s http://<export-或-内网-或-cloudflared>/v1/models | head
export VLLM_BASE=http://<同上>/v1

# ── CPU 虚拟机：环境与跑题 ──
cd ~/RL
bash scripts/setup_swebench_vm.sh
conda activate swebench

VLLM_BASE=http://127.0.0.1:8000/v1 \
SLICE=0:100 WORKERS=2 \
OUTPUT_DIR=results/swebench_vm_docker_100 \
bash scripts/run_swebench_vm_docker.sh

# ── CPU 虚拟机：评测 ──
bash scripts/eval_swebench_local.sh \
  results/swebench_vm_docker_100/preds.json \
  gemma4-vm-docker-100
```
