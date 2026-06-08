#!/usr/bin/env python3
"""Pass@1 on MBPP train split (same prompts/tests as GRPO training)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rewards.code_reward import extract_code, run_tests

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/gemma-4-E4B-it")
    parser.add_argument("--train_path", type=str, default="data/processed/mbpp_train")
    parser.add_argument("--adapter_path", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--max_lora_rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def _score_sample(completion: str, test_list: list[str], test_imports: list[str]) -> dict:
    code = extract_code(completion)
    if code is None:
        return {"passed": 0, "total": len(test_list), "pass_all": False, "has_code": False}
    passed, total = run_tests(code, test_imports, test_list)
    return {
        "passed": passed,
        "total": total,
        "pass_all": passed == total and total > 0,
        "has_code": True,
    }


def main() -> None:
    args = parse_args()
    dataset = load_from_disk(args.train_path)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts: list[str] = []
    for row in dataset:
        prompts.append(
            tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )

    llm_kwargs = {
        "model": args.model_path,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }
    lora_request = None
    if args.adapter_path:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = args.max_lora_rank
        lora_request = LoRARequest("grpo_adapter", 1, args.adapter_path)

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling, lora_request=lora_request)

    results = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        futures = {}
        for idx, output in enumerate(outputs):
            completion = output.outputs[0].text
            row = dataset[idx]
            imports = row["test_imports"] or []
            futures[
                pool.submit(_score_sample, completion, row["test_list"], imports)
            ] = (idx, row["task_id"], completion)

        for future in as_completed(futures):
            idx, task_id, completion = futures[future]
            score = future.result()
            results.append(
                {
                    "idx": idx,
                    "task_id": task_id,
                    "pass_all": score["pass_all"],
                    "passed": score["passed"],
                    "total": score["total"],
                    "has_code": score["has_code"],
                    "completion": completion,
                }
            )

    results.sort(key=lambda item: item["idx"])
    n = len(results)
    pass_at_1 = sum(item["pass_all"] for item in results) / n if n else 0.0
    mean_exec = sum(item["passed"] / item["total"] for item in results if item["total"]) / n if n else 0.0
    has_code_rate = sum(item["has_code"] for item in results) / n if n else 0.0

    summary = {
        "model_path": args.model_path,
        "adapter_path": args.adapter_path or None,
        "train_path": args.train_path,
        "num_samples": n,
        "pass_at_1": pass_at_1,
        "mean_execution_reward": mean_exec,
        "has_code_rate": has_code_rate,
        "results": results,
    }

    label = "lora" if args.adapter_path else "baseline"
    output_path = Path(args.output) if args.output else Path(f"results/mbpp_train_pass_{label}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"model={args.model_path} adapter={args.adapter_path or 'none'} samples={n}")
    print(f"pass@1={pass_at_1:.4f}  mean_execution={mean_exec:.4f}  has_code={has_code_rate:.4f}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
