#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
OUTPUT_DIR="${OUTPUT_DIR:-models/gemma-4-12B-it}"

export HF_HUB_OFFLINE=0
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate RL

echo "==> Downloading ${MODEL_ID}"
echo "    Output: ${OUTPUT_DIR}"
echo "    Mirror: ${HF_ENDPOINT}"
echo ""
echo "Note: Accept the Gemma license on HuggingFace first:"
echo "  https://huggingface.co/${MODEL_ID}"
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
from transformers import AutoProcessor

path = "${OUTPUT_DIR}"
AutoProcessor.from_pretrained(path)
print("Processor OK")
print("Download complete. Serve with:")
print("  bash scripts/serve_gemma4_12b.sh")
PY
