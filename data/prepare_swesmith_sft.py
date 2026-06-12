#!/usr/bin/env python3
"""Prepare SWE-smith expert trajectories for Gemma4/mini-swe-agent SFT."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.swe_utils import (
    estimate_message_chars,
    filter_leakage,
    load_jsonl,
    load_verified_instance_ids,
    repo_from_instance_id,
    save_json,
    save_jsonl,
    trajectory_to_sft_row,
)


def _parse_messages(raw: Any) -> list[dict[str, Any]] | None:
    if isinstance(raw, list) and raw:
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def _extract_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    for key in ("messages", "conversation", "trajectory"):
        val = record.get(key)
        parsed = _parse_messages(val)
        if parsed:
            return parsed
    info = record.get("info") or {}
    parsed = _parse_messages(info.get("messages"))
    if parsed:
        return parsed
    return None


def _is_resolved(record: dict[str, Any]) -> bool:
    for key in ("resolved", "success", "is_resolved"):
        if key in record:
            val = record[key]
            if isinstance(val, str):
                return val.lower() in {"true", "1", "yes", "resolved", "submitted"}
            return bool(val)
    info = record.get("info") or {}
    if "resolved" in info:
        return bool(info["resolved"])
    status = str(record.get("exit_status") or info.get("exit_status") or "").lower()
    return status in {"submitted", "resolved", "success", "finished"}


def _difficulty(record: dict[str, Any]) -> float | None:
    for key in ("difficulty", "difficulty_score", "score"):
        if key in record and record[key] is not None:
            return float(record[key])
    meta = record.get("metadata") or {}
    if "difficulty" in meta:
        return float(meta["difficulty"])
    return None


def load_swesmith_records(traj_dir: Path, eval_dir: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if traj_dir.is_file() and traj_dir.suffix == ".jsonl":
        return load_jsonl(traj_dir)

    patterns = ["**/*.json", "**/*.jsonl", "**/*.traj.json"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(traj_dir.glob(pat))

    resolved_ids: set[str] = set()
    if eval_dir and eval_dir.exists():
        for report in eval_dir.glob("**/*.json"):
            try:
                data = __import__("json").loads(report.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(data, dict):
                for iid, val in data.items():
                    if isinstance(val, dict) and val.get("resolved"):
                        resolved_ids.add(iid)

    for fp in files:
        if fp.suffix == ".jsonl":
            rows.extend(load_jsonl(fp))
            continue
        try:
            data = __import__("json").loads(fp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(data, list):
            rows.extend(data)
        elif isinstance(data, dict):
            rows.append(data)

    if resolved_ids:
        filtered = []
        for row in rows:
            iid = row.get("instance_id") or row.get("id")
            if iid in resolved_ids or _is_resolved(row):
                filtered.append(row)
        rows = filtered
    else:
        rows = [r for r in rows if _is_resolved(r)]

    return rows


def stratified_sample(
    rows: list[dict[str, Any]],
    *,
    n: int,
    seed: int,
    min_difficulty: float,
    max_difficulty: float,
) -> list[dict[str, Any]]:
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        iid = row.get("instance_id") or row.get("id") or "unknown"
        diff = _difficulty(row)
        if diff is not None and not (min_difficulty <= diff <= max_difficulty):
            continue
        by_repo.setdefault(repo_from_instance_id(str(iid)), []).append(row)

    rng = random.Random(seed)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    selected: list[dict[str, Any]] = []
    while len(selected) < n and repos:
        progressed = False
        for repo in repos:
            if len(selected) >= n:
                break
            pool = by_repo[repo]
            if pool:
                selected.append(pool.pop(rng.randrange(len(pool))))
                progressed = True
        if not progressed:
            break
    return selected[:n]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj_dir", type=Path, default=None, help="SWE-smith trajectories dir or jsonl")
    parser.add_argument("--eval_dir", type=Path, default=None, help="Optional eval logs for resolved filter")
    parser.add_argument("--output", type=Path, default=Path("data/sft/swesmith_train.jsonl"))
    parser.add_argument("--stats_output", type=Path, default=Path("data/sft/swesmith_stats.json"))
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--min_difficulty", type=float, default=4.0)
    parser.add_argument("--max_difficulty", type=float, default=6.0)
    parser.add_argument("--max_chars", type=int, default=120_000, help="Rough char budget (~32k tokens)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_dataset", default="SWE-bench/SWE-smith-trajectories")
    parser.add_argument("--hf_split", default="tool")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verified_ids = load_verified_instance_ids()

    if args.hf_dataset:
        from datasets import load_dataset

        ds = load_dataset(args.hf_dataset, split=args.hf_split)
        raw_rows = [dict(row) for row in ds]
        raw_rows = [r for r in raw_rows if _is_resolved(r)]
    elif args.traj_dir:
        raw_rows = load_swesmith_records(args.traj_dir, args.eval_dir)
    else:
        raise SystemExit("Provide --traj_dir or --hf_dataset")

    print(f"==> Loaded {len(raw_rows)} SWE-smith raw records")
    sampled = stratified_sample(
        raw_rows,
        n=args.max_samples,
        seed=args.seed,
        min_difficulty=args.min_difficulty,
        max_difficulty=args.max_difficulty,
    )
    print(f"==> Sampled {len(sampled)} after difficulty/repo filter")

    sft_rows: list[dict[str, Any]] = []
    skipped = {"no_messages": 0, "too_long": 0}
    for rec in sampled:
        iid = rec.get("instance_id") or rec.get("id")
        messages = _extract_messages(rec)
        if not messages or not iid:
            skipped["no_messages"] += 1
            continue
        if estimate_message_chars(messages) > args.max_chars:
            skipped["too_long"] += 1
            continue
        sft_rows.append(
            trajectory_to_sft_row(
                instance_id=str(iid),
                messages=messages,
                source="swesmith",
                metadata={"difficulty": _difficulty(rec)},
            )
        )

    sft_rows, leak_stats = filter_leakage(sft_rows, verified_ids=verified_ids)
    save_jsonl(args.output, sft_rows)

    stats = {
        "raw": len(raw_rows),
        "sampled": len(sampled),
        "written": len(sft_rows),
        "skipped": skipped,
        "leakage": leak_stats,
        "filters": {
            "min_difficulty": args.min_difficulty,
            "max_difficulty": args.max_difficulty,
            "max_chars": args.max_chars,
            "max_samples": args.max_samples,
        },
    }
    save_json(args.stats_output, stats)
    print(f"==> Wrote {args.output} ({len(sft_rows)} rows)")
    print(f"    stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
