#!/usr/bin/env python3
"""Prepare MBPP (full, official splits) for GRPO training with lm-eval-style prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rewards.mbpp_utils import build_lmeval_mbpp_prompt, normalize_test_imports


def format_sample(row: dict) -> dict:
    text = row.get("text") or row.get("prompt", "")
    user_content = build_lmeval_mbpp_prompt(text, row["test_list"])
    return {
        "task_id": row["task_id"],
        "prompt": [{"role": "user", "content": user_content}],
        "test_list": row["test_list"],
        "test_imports": normalize_test_imports(row.get("test_setup_code") or row.get("test_imports")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("google-research-datasets/mbpp", "full")

    splits = {
        "mbpp_train": dataset["train"],
        "mbpp_test": dataset["test"],
        "mbpp_validation": dataset["validation"],
        "mbpp_fewshot": dataset["prompt"],
    }

    meta = {
        "source": "google-research-datasets/mbpp full (official splits)",
        "splits": {},
        "eval_note": "Use lm_eval --tasks mbpp_instruct for test split; --tasks humaneval_instruct for HumanEval",
    }

    for name, split in splits.items():
        formatted = split.map(format_sample, remove_columns=split.column_names)
        path = output_dir / name
        formatted.save_to_disk(str(path))
        meta["splits"][name] = len(formatted)
        print(f"Saved {name}: {path} ({len(formatted)} samples)")

    (output_dir / "split_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
