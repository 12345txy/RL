#!/usr/bin/env python3
"""Prepare SWE-Gym RL train pool (optional legacy SFT supplement)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.swe_utils import (
    estimate_message_chars,
    filter_leakage,
    load_verified_instance_ids,
    save_json,
    save_jsonl,
    trajectory_to_sft_row,
)


def _extract_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    for key in ("messages", "conversation", "trajectory"):
        val = record.get(key)
        if isinstance(val, list) and val:
            return val
    return None


def _is_resolved(record: dict[str, Any]) -> bool:
    for key in ("resolved", "success"):
        if key in record:
            return bool(record[key])
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft_output", type=Path, default=Path("data/sft/swegym_openhands_supplement.jsonl"))
    parser.add_argument("--rl_output", type=Path, default=Path("data/rl/swegym_rl_train.jsonl"))
    parser.add_argument("--stats_output", type=Path, default=Path("data/rl/swegym_stats.json"))
    parser.add_argument("--traj_hf", default="SWE-Gym/OpenHands-SFT-Trajectories")
    parser.add_argument("--traj_split", default="train.success.oss")
    parser.add_argument("--gym_hf", default="SWE-Gym/SWE-Gym")
    parser.add_argument("--max_sft_chars", type=int, default=120_000)
    parser.add_argument("--lite_only", action="store_true", help="RL pool uses SWE-Gym Lite subset only")
    parser.add_argument(
        "--rl_only",
        action="store_true",
        help="Skip legacy SFT supplement; only write RL pool",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    verified_ids = load_verified_instance_ids()
    gym_ds = load_dataset(args.gym_hf, split="train")
    gym_rows = [dict(row) for row in gym_ds]

    sft_rows: list[dict[str, Any]] = []
    if not args.rl_only:
        try:
            traj_ds = load_dataset(args.traj_hf, split=args.traj_split)
            traj_records = [dict(row) for row in traj_ds]
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: could not load {args.traj_hf} split={args.traj_split}: {exc}")
            traj_records = []

        for idx, rec in enumerate(traj_records):
            if not _is_resolved(rec):
                continue
            iid = rec.get("instance_id") or rec.get("id") or f"swegym_traj_{idx:05d}"
            messages = _extract_messages(rec)
            if not messages:
                continue
            if estimate_message_chars(messages) > args.max_sft_chars:
                continue
            sft_rows.append(
                trajectory_to_sft_row(
                    instance_id=str(iid),
                    messages=messages,
                    source="swegym_openhands",
                    metadata={"traj_index": idx},
                )
            )

        sft_rows, sft_leak = filter_leakage(sft_rows, verified_ids=verified_ids)
        save_jsonl(args.sft_output, sft_rows)
    else:
        traj_records = []
        sft_leak = {"input": 0, "kept": 0, "leaked_verified": 0}

    rl_candidates = []
    for row in gym_rows:
        iid = row["instance_id"]
        rl_candidates.append(
            {
                "instance_id": iid,
                "repo": row.get("repo"),
                "problem_statement": row.get("problem_statement") or row.get("issue_body", ""),
                "base_commit": row.get("base_commit"),
                "source": "swe_gym",
            }
        )

    if args.lite_only:
        try:
            lite = load_dataset("SWE-Gym/SWE-Gym-Lite", split="train")
            lite_ids = {r["instance_id"] for r in lite}
            rl_candidates = [r for r in rl_candidates if r["instance_id"] in lite_ids]
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: SWE-Gym-Lite unavailable: {exc}")

    rl_rows, rl_leak = filter_leakage(rl_candidates, verified_ids=verified_ids)
    save_jsonl(args.rl_output, rl_rows)

    stats = {
        "sft_trajectories_input": len(traj_records),
        "sft_written": len(sft_rows),
        "sft_leakage": sft_leak,
        "rl_written": len(rl_rows),
        "rl_leakage": rl_leak,
        "lite_only": args.lite_only,
    }
    save_json(args.stats_output, stats)
    if args.rl_only:
        print(f"==> RL pool: {args.rl_output} ({len(rl_rows)} rows)")
    else:
        print(f"==> SFT supplement: {args.sft_output} ({len(sft_rows)} rows)")
        print(f"==> RL pool:         {args.rl_output} ({len(rl_rows)} rows)")


if __name__ == "__main__":
    main()
