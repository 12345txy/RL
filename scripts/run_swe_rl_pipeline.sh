#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PHASE="${PHASE:-all}"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_swe_rl_pipeline.sh

End-to-end Gemma4-12B SWE-RL pipeline.

Env:
  PHASE=all|0|1|2|3|4
  STAGE=lora|full|rl1|rl2
EOF
}

run_phase0() {
  bash scripts/setup_swebench_vm.sh
  SKIP_VLLM_CHECK="${SKIP_VLLM_CHECK:-1}" SKIP_TOOLCHECK="${SKIP_TOOLCHECK:-1}" \
    bash scripts/run_miniswe_baseline.sh
}

run_phase1() {
  bash scripts/run_prepare_data.sh
}

run_phase2() {
  STAGE="${STAGE:-lora}" bash scripts/run_sft.sh
  STAGE=full bash scripts/run_sft.sh
}

run_phase3() {
  STAGE=rl1 bash scripts/run_rl_skyrl.sh
  STAGE=rl2 bash scripts/run_rl_skyrl.sh
}

run_phase4() {
  bash scripts/run_verifier.sh
}

case "${PHASE}" in
  all)
    run_phase0
    run_phase1
    run_phase2
    run_phase3
    run_phase4
    ;;
  0) run_phase0 ;;
  1) run_phase1 ;;
  2) run_phase2 ;;
  3) run_phase3 ;;
  4) run_phase4 ;;
  *) usage; exit 1 ;;
esac
