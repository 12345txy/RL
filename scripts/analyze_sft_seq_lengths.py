#!/usr/bin/env python3
"""Analyze SFT token lengths and recommend max_seq_length."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.swe_utils import load_jsonl, prepare_gemma4_chat_messages, save_json


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * pct / 100.0)) - 1))
    return ordered[idx]


def recommend_max_seq_length(
    lengths: list[int],
    *,
    percentile: float,
    block_size: int,
    max_cap: int,
) -> int:
    target = _percentile(lengths, percentile)
    recommended = math.ceil(target / block_size) * block_size
    recommended = max(block_size, recommended)
    return min(max_cap, recommended)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="models/gemma-4-12B-it")
    parser.add_argument("--train_path", default="data/sft/sft_merged.jsonl")
    parser.add_argument("--output_path", default="data/sft/seq_length_stats.json")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--block_size", type=int, default=4096)
    parser.add_argument("--max_cap", type=int, default=32768)
    parser.add_argument("--compare_caps", default="24576,28672,32768,36864,40960")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    rows = load_jsonl(args.train_path)
    if not rows:
        raise SystemExit(f"No rows in {args.train_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    lengths: list[int] = []
    for row in rows:
        messages = prepare_gemma4_chat_messages(row["messages"])
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))

    compare_caps = [int(x.strip()) for x in args.compare_caps.split(",") if x.strip()]
    truncation_at_cap = {
        str(cap): sum(1 for length in lengths if length > cap) for cap in compare_caps
    }
    recommended = recommend_max_seq_length(
        lengths,
        percentile=args.percentile,
        block_size=args.block_size,
        max_cap=args.max_cap,
    )
    stats = {
        "model_path": args.model_path,
        "train_path": args.train_path,
        "num_samples": len(lengths),
        "percentiles": {
            "p50": _percentile(lengths, 50),
            "p90": _percentile(lengths, 90),
            "p95": _percentile(lengths, 95),
            "p99": _percentile(lengths, 99),
            "max": max(lengths),
        },
        "percentile_used": args.percentile,
        "block_size": args.block_size,
        "max_cap": args.max_cap,
        "recommended_max_seq_length": recommended,
        "truncation_at_cap": truncation_at_cap,
        "truncation_pct_at_recommended": round(
            100.0 * sum(1 for length in lengths if length > recommended) / len(lengths),
            2,
        ),
    }
    save_json(args.output_path, stats)
    print(f"==> Analyzed {len(lengths)} samples -> {args.output_path}")
    print(
        "    p50={p50} p90={p90} p95={p95} max={max} recommended={rec} "
        "(trunc={trunc}%)".format(
            **stats["percentiles"],
            rec=recommended,
            trunc=stats["truncation_pct_at_recommended"],
        )
    )


if __name__ == "__main__":
    main()
