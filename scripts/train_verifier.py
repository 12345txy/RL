#!/usr/bin/env python3
"""Train a lightweight trajectory verifier for Best-of-N selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.swe_utils import load_jsonl


class VerifierDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int = 4096):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        enc["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return enc


def trajectory_to_text(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"[{role}]\n{content}")
    return "\n\n".join(parts)


def build_verifier_rows(paths: list[str], max_samples: int | None) -> list[dict]:
    rows_raw: list[dict] = []
    for path in paths:
        rows_raw.extend(load_jsonl(path))

    rows: list[dict] = []
    for rec in rows_raw:
        messages = rec.get("messages")
        if not messages:
            continue
        label = 1 if rec.get("metadata", {}).get("reward", 0) >= 1.0 else 0
        if "label" in rec:
            label = int(rec["label"])
        elif rec.get("source") in {"swesmith", "swegym_openhands"}:
            label = 1
        rows.append({"text": trajectory_to_text(messages), "label": label})
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, help="Rollout jsonl with reward/resolved labels")
    parser.add_argument("--base_model", default="microsoft/deberta-v3-base")
    parser.add_argument("--output_dir", default="outputs/verifier-gemma4-12b")
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_verifier_rows(args.input, args.max_samples)
    if len(rows) < 10:
        raise SystemExit(f"Need >=10 verifier rows, got {len(rows)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    ds = VerifierDataset(rows, tokenizer, max_length=args.max_length)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=10,
        save_steps=200,
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=ds)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"==> Verifier saved to {args.output_dir}")


if __name__ == "__main__":
    main()
