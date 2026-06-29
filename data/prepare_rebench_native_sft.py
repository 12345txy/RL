#!/usr/bin/env python3
"""Prepare SWE-ReBench native-tool mini-swe-agent v2 trajectories for Gemma4 SFT."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.swe_utils import (
    count_chat_tokens,
    estimate_message_chars,
    estimate_message_tokens,
    filter_leakage,
    load_jsonl,
    load_verified_instance_ids,
    looks_like_native_tool_format,
    save_json,
    save_jsonl,
    trajectory_to_sft_row,
)


def _extract_instance_id(record: dict[str, Any]) -> str | None:
    for key in ("instance_id", "id"):
        if record.get(key):
            return str(record[key])
    meta = record.get("metadata") or {}
    if meta.get("instance_id"):
        return str(meta["instance_id"])
    sample_id = meta.get("sample_id")
    if sample_id:
        text = str(sample_id)
        if "::" in text:
            return text.split("::", 1)[1]
        return text
    return None


def _extract_messages(record: dict[str, Any]) -> list[dict[str, Any]] | None:
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return messages
    return None


def _is_resolved(record: dict[str, Any]) -> bool:
    meta = record.get("metadata") or {}
    for container in (record, meta):
        for key in ("resolved", "success", "is_resolved"):
            if key in container:
                val = container[key]
                if isinstance(val, str):
                    return val.lower() in {"true", "1", "yes", "resolved", "submitted"}
                return bool(val)
        status = str(container.get("exit_status") or container.get("verifier_status") or "").lower()
        if status in {"submitted", "resolved", "success", "finished", "completed"}:
            return True
    return True


def _validate_native_tool_turns(messages: list[dict[str, Any]]) -> str | None:
    if not looks_like_native_tool_format(messages):
        return "not_native_tool_format"
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return "assistant_missing_tool_calls"
        if len(tool_calls) != 1:
            return "assistant_multiple_tool_calls"
        fn = (tool_calls[0].get("function") or {}) if isinstance(tool_calls[0], dict) else {}
        if fn.get("name") != "bash":
            return "unexpected_tool_name"
        args = fn.get("arguments")
        if not isinstance(args, (dict, str)):
            return "invalid_tool_arguments"
        if isinstance(args, dict) and not args.get("command"):
            return "missing_bash_command"
    return None


def _message_token_count(messages: list[dict[str, Any]], tokenizer=None) -> int:
    if tokenizer is not None:
        return count_chat_tokens(messages, tokenizer)
    return estimate_message_tokens(messages)


def records_to_sft_rows(
    records: list[dict[str, Any]],
    *,
    source: str,
    max_chars: int,
    filter_max_tokens: int,
    tokenizer=None,
    resolved_only: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sft_rows: list[dict[str, Any]] = []
    skipped = {
        "no_messages": 0,
        "no_instance_id": 0,
        "unresolved": 0,
        "too_long": 0,
        "too_many_tokens": 0,
        "invalid_native_tool": 0,
    }
    invalid_reasons: dict[str, int] = {}

    for rec in records:
        if resolved_only and not _is_resolved(rec):
            skipped["unresolved"] += 1
            continue
        messages = _extract_messages(rec)
        if not messages:
            skipped["no_messages"] += 1
            continue
        iid = _extract_instance_id(rec)
        if not iid:
            skipped["no_instance_id"] += 1
            continue
        reason = _validate_native_tool_turns(messages)
        if reason:
            skipped["invalid_native_tool"] += 1
            invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            continue
        if max_chars > 0 and estimate_message_chars(messages) > max_chars:
            skipped["too_long"] += 1
            continue
        if filter_max_tokens > 0 and _message_token_count(messages, tokenizer) > filter_max_tokens:
            skipped["too_many_tokens"] += 1
            continue

        meta = dict(rec.get("metadata") or {})
        sft_rows.append(
            trajectory_to_sft_row(
                instance_id=iid,
                messages=messages,
                source=source,
                metadata={
                    "resolved": _is_resolved(rec),
                    "dataset": meta.get("dataset", "swe-rebench-v2"),
                    "teacher": meta.get("teacher"),
                    "scaffold": meta.get("scaffold", "mini-swe-agent-native-tool"),
                    "n_calls": meta.get("n_calls"),
                    "exit_status": meta.get("exit_status"),
                },
            )
        )

    if invalid_reasons:
        skipped["invalid_reasons"] = invalid_reasons  # type: ignore[assignment]
    return sft_rows, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records_path",
        type=Path,
        default=Path("rebench_native_tool_clean_resolved_repaired_225/records.jsonl"),
        help="records.jsonl with messages + metadata (preferred)",
    )
    parser.add_argument(
        "--train_path",
        type=Path,
        default=None,
        help="Fallback train.jsonl when records_path is missing",
    )
    parser.add_argument("--output", type=Path, default=Path("data/sft/rebench_native_train.jsonl"))
    parser.add_argument("--stats_output", type=Path, default=Path("data/sft/rebench_native_stats.json"))
    parser.add_argument(
        "--max_chars",
        type=int,
        default=0,
        help="Drop trajectories longer than this many chars; 0 keeps all rows",
    )
    parser.add_argument(
        "--filter_max_tokens",
        type=int,
        default=200_000,
        help="Drop extreme trajectories above this token count; 0 disables",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/gemma-4-12B-it",
        help="Tokenizer path for exact token filtering when filter_max_tokens > 0",
    )
    parser.add_argument(
        "--no_exact_token_filter",
        action="store_true",
        help="Use char/token estimate instead of loading the tokenizer during prep",
    )
    parser.add_argument(
        "--resolved_only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--source", default="rebench_native_tool")
    return parser.parse_args()


def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.records_path.is_file():
        return load_jsonl(args.records_path)
    if args.train_path and args.train_path.is_file():
        return load_jsonl(args.train_path)
    if args.train_path is None:
        fallback = args.records_path.parent / "train.jsonl"
        if fallback.is_file():
            print(f"WARN: {args.records_path} missing; falling back to {fallback}")
            return load_jsonl(fallback)
    raise SystemExit(f"No input found: {args.records_path}")


def main() -> None:
    args = parse_args()
    records = _load_records(args)
    print(f"==> Loaded {len(records)} native-tool records from {args.records_path}")

    tokenizer = None
    if args.filter_max_tokens > 0 and not args.no_exact_token_filter:
        model_path = Path(args.model_path)
        if model_path.is_dir():
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
            print(f"==> Using exact token filter with {model_path}")
        else:
            print(f"WARN: model_path missing ({model_path}); using estimated token filter")

    sft_rows, skipped = records_to_sft_rows(
        records,
        source=args.source,
        max_chars=args.max_chars,
        filter_max_tokens=args.filter_max_tokens,
        tokenizer=tokenizer,
        resolved_only=args.resolved_only,
    )

    verified_ids = load_verified_instance_ids()
    sft_rows, leak_stats = filter_leakage(sft_rows, verified_ids=verified_ids)

    save_jsonl(args.output, sft_rows)
    stats = {
        "records_path": str(args.records_path),
        "raw": len(records),
        "written": len(sft_rows),
        "skipped": skipped,
        "leakage": leak_stats,
        "filters": {
            "max_chars": args.max_chars,
            "filter_max_tokens": args.filter_max_tokens,
            "exact_token_filter": tokenizer is not None,
            "resolved_only": args.resolved_only,
            "source": args.source,
        },
    }
    save_json(args.stats_output, stats)
    print(f"==> Wrote {args.output} ({len(sft_rows)} rows)")
    print(f"    skipped={skipped}")
    print(f"    leakage={leak_stats}")
    print(f"    stats -> {args.stats_output}")


if __name__ == "__main__":
    main()
