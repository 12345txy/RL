#!/usr/bin/env python3
"""Best-of-N SWE-bench evaluation with a trained verifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.swe_utils import load_json, save_json
from integrations.skyrl_miniswe.generator import MiniSweRolloutGenerator
from scripts.train_verifier import trajectory_to_text


def load_instances(dev_split: str | None, subset: str, split: str) -> list[dict]:
    from datasets import load_dataset

    if dev_split:
        payload = load_json(dev_split)
        allowed = set(payload.get("instance_ids", payload))
    else:
        allowed = None

    ds_name = "princeton-nlp/SWE-bench_Verified" if subset == "verified" else "princeton-nlp/SWE-bench_Lite"
    ds = load_dataset(ds_name, split=split)
    rows = [dict(r) for r in ds]
    if allowed is not None:
        rows = [r for r in rows if r["instance_id"] in allowed]
    return rows


def score_trajectory(verifier, tokenizer, messages: list[dict]) -> float:
    text = trajectory_to_text(messages)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    enc = {k: v.to(verifier.device) for k, v in enc.items()}
    with torch.no_grad():
        logits = verifier(**enc).logits
        probs = torch.softmax(logits, dim=-1)
    return float(probs[0, 1].item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("results/verified_final"))
    parser.add_argument("--vllm_base", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="gemma-4-12B-it")
    parser.add_argument("--verifier", required=True)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--subset", default="verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--dev_split", default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    instances = load_instances(args.dev_split, args.subset, args.split)
    generator = MiniSweRolloutGenerator(
        api_base=args.vllm_base,
        model=args.model,
        max_turns=50,
        temperature=args.temperature,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.verifier, trust_remote_code=True)
    verifier = AutoModelForSequenceClassification.from_pretrained(
        args.verifier, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).eval()
    if torch.cuda.is_available():
        verifier = verifier.cuda()

    preds: dict[str, dict] = {}
    for inst in instances:
        iid = inst["instance_id"]
        best_score = -1.0
        best_patch = ""
        for _ in range(args.k):
            rollout = generator.rollout(inst)
            score = score_trajectory(verifier, tokenizer, rollout.messages)
            if score > best_score:
                best_score = score
                best_patch = rollout.patch
        preds[iid] = {
            "model_name_or_path": args.model,
            "instance_id": iid,
            "model_patch": best_patch,
            "verifier_score": best_score,
        }
        print(f"{iid} best_score={best_score:.4f} patch_len={len(best_patch)}")

    save_json(args.output_dir / "preds.json", preds)
    save_json(
        args.output_dir / "run_metadata.json",
        {"k": args.k, "n_instances": len(instances), "verifier": args.verifier},
    )
    print(f"==> Wrote {args.output_dir / 'preds.json'}")


if __name__ == "__main__":
    main()
