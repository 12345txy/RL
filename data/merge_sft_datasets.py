#!/usr/bin/env python3
"""Merge SWE-smith + SWE-Gym SFT JSONL with stratified shuffle."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.swe_utils import load_jsonl, repo_from_instance_id, save_json, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSONL files")
    parser.add_argument("--output", type=Path, default=Path("data/sft/sft_merged.jsonl"))
    parser.add_argument("--stats_output", type=Path, default=Path("data/sft/sft_merged_stats.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gym_weight", type=float, default=0.2, help="Target fraction for swegym source")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.inputs:
        part = load_jsonl(path)
        print(f"    {path}: {len(part)} rows")
        rows.extend(part)

    by_source: dict[str, list] = {}
    for row in rows:
        by_source.setdefault(row.get("source", "unknown"), []).append(row)

    rng = random.Random(args.seed)
    merged: list = []
    gym = by_source.get("swegym_openhands", [])
    smith = by_source.get("swesmith", [])
    miniswe = [
        r
        for src, items in by_source.items()
        if src.startswith("miniswe") or src == "miniswe"
        for r in items
    ]
    other = [
        r
        for src, items in by_source.items()
        if src not in {"swegym_openhands", "swesmith"} and not src.startswith("miniswe") and src != "miniswe"
        for r in items
    ]

    if miniswe and not smith and not gym:
        merged.extend(miniswe)
    elif smith and gym:
        target_total = len(smith) + len(gym)
        target_gym = max(len(gym), int(target_total * args.gym_weight))
        if len(gym) < target_gym:
            extra = [gym[rng.randrange(len(gym))] for _ in range(target_gym - len(gym))]
            gym = gym + extra
        merged.extend(smith)
        merged.extend(gym[:target_gym])
    else:
        merged.extend(smith or gym or other)

    merged.extend(other)
    rng.shuffle(merged)

    save_jsonl(args.output, merged)
    stats = {
        "inputs": args.inputs,
        "total": len(merged),
        "source_counts": dict(Counter(r.get("source", "unknown") for r in merged)),
        "repo_counts": dict(Counter(repo_from_instance_id(r["instance_id"]) for r in merged)),
        "seed": args.seed,
    }
    save_json(args.stats_output, stats)
    print(f"==> Wrote {args.output} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
