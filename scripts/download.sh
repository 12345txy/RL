#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
OUTPUT_DIR="${OUTPUT_DIR:-models/Qwen3.6-27B}"

export HF_HUB_OFFLINE=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

echo "==> Downloading ${MODEL_ID}"
echo "    Output: ${OUTPUT_DIR}"
echo "    Mirror: ${HF_ENDPOINT}"
echo ""
echo "Model page: https://huggingface.co/${MODEL_ID}"
echo "License: Apache 2.0 (resume with snapshot_download if interrupted)"
echo ""

python - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${OUTPUT_DIR}",
    local_dir_use_symlinks=False,
)
print("Done:", "${OUTPUT_DIR}")
PY

echo "==> Verify load"
python - <<PY
from transformers import AutoTokenizer

path = "${OUTPUT_DIR}"
AutoTokenizer.from_pretrained(path, trust_remote_code=True)
print("Tokenizer OK")
print("Download complete:", path)
PY
