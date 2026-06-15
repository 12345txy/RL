#!/usr/bin/env python3
"""Convert SWE-Gym instances to SkyRL parquet format (train + validation)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.swe_utils import filter_leakage, load_verified_instance_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="data/rl/skyrl_parquet")
    parser.add_argument("--gym_hf", default="SWE-Gym/SWE-Gym")
    parser.add_argument("--val_hf", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--val_split", default="test")
    parser.add_argument("--lite_only", action="store_true", help="Train on SWE-Gym-Lite subset only")
    parser.add_argument("--max_train", type=int, default=0, help="Cap train rows (0 = all)")
    parser.add_argument("--max_val", type=int, default=50, help="Cap validation rows")
    return parser.parse_args()


def _to_skyrl_row(example: dict, data_source: str) -> dict:
    problem = example.get("problem_statement") or example.get("issue_body") or ""
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": problem}],
        "env_class": "null",
        "instance": example,
    }


def main() -> None:
    args = parse_args()
    from datasets import load_dataset

    output_dir = Path(os.path.expanduser(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    verified_ids = load_verified_instance_ids(split=args.val_split)
    train_ds = load_dataset(args.gym_hf, split="train")
    train_rows = [dict(row) for row in train_ds]

    if args.lite_only:
        try:
            lite = load_dataset("SWE-Gym/SWE-Gym-Lite", split="train")
            lite_ids = {r["instance_id"] for r in lite}
            train_rows = [r for r in train_rows if r["instance_id"] in lite_ids]
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: SWE-Gym-Lite unavailable: {exc}")

    train_candidates = []
    for row in train_rows:
        skyrl_row = _to_skyrl_row(row, args.gym_hf)
        skyrl_row["instance_id"] = row["instance_id"]
        train_candidates.append(skyrl_row)

    train_filtered, train_leak = filter_leakage(train_candidates, verified_ids=verified_ids)

    if args.max_train > 0:
        train_filtered = train_filtered[: args.max_train]

    val_ds = load_dataset(args.val_hf, split=args.val_split)
    val_rows = [_to_skyrl_row(dict(row), args.val_hf) for row in val_ds]
    if args.max_val > 0:
        val_rows = val_rows[: args.max_val]
    for row in val_rows:
        row["instance_id"] = row["instance"]["instance_id"]

    import datasets as hf_datasets

    train_out = output_dir / ("train_lite.parquet" if args.lite_only else "train.parquet")
    val_out = output_dir / "validation.parquet"
    hf_datasets.Dataset.from_list(train_filtered).to_parquet(str(train_out))
    hf_datasets.Dataset.from_list(val_rows).to_parquet(str(val_out))

    print(f"==> Wrote {train_out} ({len(train_filtered)} rows, leakage={train_leak})")
    print(f"==> Wrote {val_out} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
