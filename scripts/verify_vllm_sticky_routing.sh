#!/usr/bin/env bash
# Verify nginx sticky routing: same X-SWE-Instance-Id → same vLLM backend.
set -euo pipefail

NGINX_URL="${NGINX_URL:-http://127.0.0.1:8001}"
INSTANCE_ID="${INSTANCE_ID:-django__django-sticky-probe}"

pick_backend() {
  curl -sI "${NGINX_URL%/}/v1/models" -H "X-SWE-Instance-Id: $1" \
    | awk -F': ' 'tolower($1)=="x-vllm-backend" {print $2}' | tr -d '\r'
}

b1="$(pick_backend "$INSTANCE_ID")"
b2="$(pick_backend "$INSTANCE_ID")"
echo "backend1=$b1"
echo "backend2=$b2"

if [[ -z "$b1" || -z "$b2" ]]; then
  echo "FAIL: missing X-VLLM-Backend header (reload nginx with scripts/nginx_vllm_lb.sh reload)" >&2
  exit 1
fi
if [[ "$b1" != "$b2" ]]; then
  echo "FAIL: sticky routing broken" >&2
  exit 1
fi
echo "PASS: sticky routing OK ($INSTANCE_ID -> $b1)"
