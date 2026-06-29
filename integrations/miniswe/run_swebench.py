#!/usr/bin/env python3
"""mini-extra swebench with nginx sticky vLLM routing (X-SWE-Instance-Id)."""

from __future__ import annotations

import sys

from integrations.miniswe.sticky_routing import apply_sticky_vllm_routing


def main() -> None:
    apply_sticky_vllm_routing()
    from minisweagent.run.benchmarks.swebench import app

    app(sys.argv[1:], prog_name="mini-extra swebench")


if __name__ == "__main__":
    main()
