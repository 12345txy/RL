#!/usr/bin/env python3
"""Prepare mini-swe-agent trajectories for Gemma4 SFT."""

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
        parsed = _parse_messages(record.get(key))
        if parsed:
            return parsed
    info = record.get("info") or {}
    parsed = _parse_messages(info.get("messages"))
    if parsed:
        return parsed
    nested = record.get("trajectory")
    if isinstance(nested, dict):
        parsed = _parse_messages(nested.get("messages"))
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
    if status:
        return status in {"submitted", "resolved", "success", "finished"}
    return True


def _looks_like_miniswe(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        content = msg.get("content") or ""
        if msg.get("role") == "assistant" and "THOUGHT:" in content:
            return True
        if msg.get("role") == "user" and "<returncode>" in content:
            return True
    return False


def load_local_records(traj_dir: Path) -> list[dict[str, Any]]:
    if traj_dir.is_file() and traj_dir.suffix == ".jsonl":
        return load_jsonl(traj_dir)

    rows: list[dict[str, Any]] = []
    for pat in ("**/*.jsonl", "**/*.traj.json", "**/*.json"):
        for fp in traj_dir.glob(pat):
            if fp.suffix == ".jsonl":
                rows.extend(load_jsonl(fp))
                continue
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(data, list):
                rows.extend(data)
            elif isinstance(data, dict):
                rows.append(data)
    return rows


def load_hf_records(dataset_id: str, split: str) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    return [dict(row) for row in ds]


def stratified_sample(rows: list[dict[str, Any]], *, n: int, seed: int) -> list[dict[str, Any]]:
    if n <= 0 or len(rows) <= n:
        return rows

    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        iid = row.get("instance_id") or row.get("id") or "unknown"
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


def records_to_sft_rows(
    records: list[dict[str, Any]],
    *,
    source: str,
    max_chars: int,
    resolved_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sft_rows: list[dict[str, Any]] = []
    skipped = {
        "no_messages": 0,
        "no_instance_id": 0,
        "unresolved": 0,
        "too_long": 0,
        "not_miniswe_format": 0,
    }
    for rec in records:
        if resolved_only and not _is_resolved(rec):
            skipped["unresolved"] += 1
            continue
        iid = rec.get("instance_id") or rec.get("id")
        messages = _extract_messages(rec)
        if not messages:
            skipped["no_messages"] += 1
            continue
        if not iid:
            skipped["no_instance_id"] += 1
            continue
        if not _looks_like_miniswe(messages):
            skipped["not_miniswe_format"] += 1
            continue
        if estimate_message_chars(messages) > max_chars:
            skipped["too_long"] += 1
            continue
        sft_rows.append(
            trajectory_to_sft_row(
                instance_id=str(iid),
                messages=messages,
                source=source,
                metadata={
                    "exit_status": rec.get("exit_status"),
                    "resolved": rec.get("resolved", _is_resolved(rec)),
                    "n_turns": rec.get("n_turns"),
                },
            )
        )
    return sft_rows, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf_dataset",
        default="Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k",
        help="Primary HF dataset (mini-swe-agent / mini-swe-agent-plus trajectories)",
    )
    parser.add_argument("--hf_split", default="train")
    parser.add_argument(
        "--extra_hf_dataset",
        default="JetBrains-Research/agent-trajectories-swesmith-random-subset",
        help="Optional second HF dataset; set empty string to skip",
    )
    parser.add_argument("--extra_hf_split", default="train")
    parser.add_argument("--traj_dir", type=Path, default=None, help="Optional local traj dir/jsonl")
    parser.add_argument("--output", type=Path, default=Path("data/sft/miniswe_train.jsonl"))
    parser.add_argument("--stats_output", type=Path, default=Path("data/sft/miniswe_stats.json"))
    parser.add_argument("--max_samples", type=int, default=5000)
    parser.add_argument("--extra_max_samples", type=int, default=1500)
    parser.add_argument("--max_chars", type=int, default=120_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resolved_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep resolved/submitted trajectories only when the field exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verified_ids = load_verified_instance_ids()

    primary_rows: list[dict[str, Any]] = []
    if args.traj_dir:
        primary_rows = load_local_records(args.traj_dir)
        primary_source = "miniswe_local"
        print(f"==> Loaded {len(primary_rows)} local mini-swe records from {args.traj_dir}")
    elif args.hf_dataset:
        primary_rows = load_hf_records(args.hf_dataset, args.hf_split)
        primary_source = "miniswe_kwai" if "kwai" in args.hf_dataset.lower() else "miniswe_hf"
        print(f"==> Loaded {len(primary_rows)} HF records from {args.hf_dataset} [{args.hf_split}]")
    else:
        raise SystemExit("Provide --hf_dataset or --traj_dir")

    primary_sampled = stratified_sample(primary_rows, n=args.max_samples, seed=args.seed)
    print(f"==> Sampled {len(primary_sampled)} primary rows (max_samples={args.max_samples})")

    sft_rows, skipped_primary = records_to_sft_rows(
        primary_sampled,
        source=primary_source,
        max_chars=args.max_chars,
        resolved_only=args.resolved_only,
    )

    extra_stats: dict[str, Any] = {}
    if args.extra_hf_dataset:
        extra_rows = load_hf_records(args.extra_hf_dataset, args.extra_hf_split)
        extra_sampled = stratified_sample(extra_rows, n=args.extra_max_samples, seed=args.seed + 1)
        extra_sft, skipped_extra = records_to_sft_rows(
            extra_sampled,
            source="miniswe_jetbrains",
            max_chars=args.max_chars,
            resolved_only=True,
        )
        print(
            f"==> Loaded {len(extra_rows)} extra HF records from {args.extra_hf_dataset}; "
            f"kept {len(extra_sft)} after filters"
        )
        sft_rows.extend(extra_sft)
        extra_stats = {
            "dataset": args.extra_hf_dataset,
            "raw": len(extra_rows),
            "sampled": len(extra_sampled),
            "written": len(extra_sft),
            "skipped": skipped_extra,
        }

    sft_rows, leak_stats = filter_leakage(sft_rows, verified_ids=verified_ids)
    save_jsonl(args.output, sft_rows)

    stats = {
        "primary_dataset": args.hf_dataset or str(args.traj_dir),
        "primary_raw": len(primary_rows),
        "primary_sampled": len(primary_sampled),
        "written": len(sft_rows),
        "skipped_primary": skipped_primary,
        "extra": extra_stats,
        "leakage": leak_stats,
        "filters": {
            "max_samples": args.max_samples,
            "extra_max_samples": args.extra_max_samples,
            "max_chars": args.max_chars,
            "resolved_only": args.resolved_only,
            "seed": args.seed,
        },
        "source_counts": {},
    }
    from collections import Counter

    stats["source_counts"] = dict(Counter(r.get("source", "unknown") for r in sft_rows))
    save_json(args.stats_output, stats)
    print(f"==> Wrote {args.output} ({len(sft_rows)} rows)")
    print(f"    sources: {stats['source_counts']}")
    print(f"    stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
