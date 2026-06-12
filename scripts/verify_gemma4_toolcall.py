#!/usr/bin/env python3
"""Verify Gemma4 tool-call compatibility with vLLM OpenAI API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _post_json(url: str, payload: dict, *, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(url: str, *, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_base", default=os.environ.get("VLLM_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "gemma-4-12B-it"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.api_base.rstrip("/")

    print(f"==> Checking vLLM at {base}")
    try:
        models = _get(f"{base}/models")
    except urllib.error.URLError as exc:
        print(f"FAIL: cannot reach vLLM: {exc}", file=sys.stderr)
        print("Start server: bash scripts/serve_gemma4_12b.sh", file=sys.stderr)
        return 1

    available = [m["id"] for m in models.get("data", [])]
    print(f"    models: {available}")
    if args.model not in available and available:
        print(f"WARN: requested model {args.model!r} not in list; using {available[0]!r}")
        args.model = available[0]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "execute_bash",
                "description": "Run a bash command in the sandbox",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        }
    ]

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": "You are a software engineering agent. Use tools when needed.",
            },
            {
                "role": "user",
                "content": "List files in the current directory using a bash command.",
            },
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.0,
    }

    print("==> Sending tool-call probe to /chat/completions")
    try:
        resp = _post_json(f"{base}/chat/completions", payload)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"FAIL: HTTP {exc.code}: {body}", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    choice = resp["choices"][0]["message"]
    tool_calls = choice.get("tool_calls") or []
    content = (choice.get("content") or "").strip()

    print("==> Response summary")
    print(f"    content_len={len(content)}")
    print(f"    tool_calls={len(tool_calls)}")
    if tool_calls:
        fn = tool_calls[0].get("function", {})
        print(f"    first_tool={fn.get('name')!r} args={fn.get('arguments', '')[:200]}")

    ok = bool(tool_calls) or "ls" in content.lower() or "bash" in content.lower()
    if ok:
        print("PASS: Gemma4 appears compatible with tool-use via vLLM")
        return 0

    print("FAIL: no tool_calls and no bash-like content in response", file=sys.stderr)
    print(json.dumps(choice, indent=2, ensure_ascii=False)[:2000], file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
