#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

echo "==> Installing SWE-bench cloud dependencies in RL env..."
python -m pip install -U pip
python -m pip install -r requirements.txt -r requirements-swebench.txt

echo
echo "==> Installed commands:"
command -v mini-extra
command -v sb-cli
command -v modal

echo
echo "==> One-time authentication (required before first run)"
echo
echo "[1] Modal (agent sandbox on cloud)"
echo "    modal setup"
echo "    # Follow browser login; verify with: modal profile list"
echo
echo "[2] sb-cli (cloud evaluation)"
echo "    sb-cli gen-api-key your.email@example.com"
echo "    export SWEBENCH_API_KEY=<key_from_email>"
echo "    sb-cli verify-api-key <verification_code>"
echo
echo "[3] Model API"
echo "    Option A - local vLLM (default in run script):"
echo "      vllm serve <model_path> --port 8000"
echo "    Option B - cloud API model:"
echo "      export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY=..."
echo "      # then pass --model openai/gpt-4o (etc.) to run_swebench_cloud.sh"
echo
echo "Setup complete. Next:"
echo "  bash scripts/run_swebench_cloud.sh"
echo "  bash scripts/eval_swebench_sbcli.sh <preds.json> <run_id>"
