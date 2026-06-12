# Sky-RL + mini-swe-agent integration

Rollout generator and reward helpers for multi-turn SWE Agent RL with vLLM.

## Usage

```bash
pip install git+https://github.com/NovaSky-AI/SkyRL.git
```

Then map `integrations/skyrl_miniswe/config.yaml` into your Sky-RL launch script.

## Components

| File | Role |
|------|------|
| `generator.py` | Multi-turn mini-swe-agent rollout via vLLM `/chat/completions` |
| `reward.py` | Binary resolve reward for SWE-Gym RL |
| `config.yaml` | Reference Sky-RL trainer config |

## Serve checkpoint

```bash
CHECKPOINT=outputs/sft-gemma4-12b-miniswe-full bash scripts/serve_gemma4_12b.sh
```

## SWE-bench eval

```bash
VLLM_BASE=http://127.0.0.1:8000/v1 SLICE=0:100 WORKERS=2 \
  bash scripts/run_swebench_vm_docker.sh
```
