"""Sticky vLLM routing: attach X-SWE-Instance-Id on every LLM call per SWE-bench instance."""

from __future__ import annotations

import copy
from typing import Any

from minisweagent.run.benchmarks import swebench as swebench_module

SWE_INSTANCE_HEADER = "X-SWE-Instance-Id"
_PATCHED = False


def _inject_instance_header(config: dict[str, Any], instance_id: str) -> dict[str, Any]:
    config = copy.deepcopy(config)
    model_cfg = config.setdefault("model", {})
    model_kwargs = model_cfg.setdefault("model_kwargs", {})
    extra_headers = dict(model_kwargs.get("extra_headers") or {})
    extra_headers[SWE_INSTANCE_HEADER] = instance_id
    model_kwargs["extra_headers"] = extra_headers
    return config


def _patch_process_instance(module: Any) -> None:
    original = module.process_instance

    def process_instance(instance, output_dir, config, progress_manager):
        instance_id = instance["instance_id"]
        config = _inject_instance_header(config, instance_id)
        return original(instance, output_dir, config, progress_manager)

    module.process_instance = process_instance


def apply_sticky_vllm_routing() -> None:
    """Monkey-patch mini-swe-agent SWE-bench batch runner to send sticky routing headers."""
    global _PATCHED
    if _PATCHED:
        return
    _patch_process_instance(swebench_module)
    _PATCHED = True
