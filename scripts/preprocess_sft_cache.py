#!/usr/bin/env python3
"""Build on-disk SFT preprocess cache without training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from training.sft_preprocess_cache import is_main_process, prepare_sft_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="models/gemma-4-12B-it")
    parser.add_argument("--train_path", default="data/sft/sft_merged.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=28672)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--preprocess_cache_dir", default="data/sft/preprocessed")
    parser.add_argument("--force_preprocess", action="store_true")
    parser.add_argument("--preprocess_num_proc", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _, cache_dir, from_cache = prepare_sft_dataset(
        train_path=args.train_path,
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        cache_base_dir=args.preprocess_cache_dir,
        use_preprocessed=not args.force_preprocess,
        force_preprocess=args.force_preprocess,
        preprocess_num_proc=args.preprocess_num_proc,
        is_main_process=is_main_process(),
    )
    if is_main_process():
        action = "loaded" if from_cache else "built"
        print(f"==> Preprocess cache {action}: {cache_dir}")


if __name__ == "__main__":
    main()
