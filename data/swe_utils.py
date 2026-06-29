"""Shared utilities for SWE SFT/RL data preparation."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERIFIED_DATASET = "princeton-nlp/SWE-bench_Verified"
SWE_GYM_DATASET = "SWE-Gym/SWE-Gym"


def load_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_verified_instance_ids(*, split: str = "test") -> set[str]:
    from datasets import load_dataset

    ds = load_dataset(VERIFIED_DATASET, split=split)
    return {row["instance_id"] for row in ds}


def load_swe_gym_instance_ids() -> set[str]:
    from datasets import load_dataset

    ds = load_dataset(SWE_GYM_DATASET, split="train")
    return {row["instance_id"] for row in ds}


def repo_from_instance_id(instance_id: str) -> str:
    return instance_id.rsplit("-", 1)[0]


def make_verified_dev_split(
    *,
    n: int = 100,
    seed: int = 42,
    split: str = "test",
    output_path: str | Path | None = None,
) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(VERIFIED_DATASET, split=split)
    by_repo: dict[str, list[str]] = defaultdict(list)
    for row in ds:
        by_repo[repo_from_instance_id(row["instance_id"])].append(row["instance_id"])

    rng = random.Random(seed)
    selected: list[str] = []
    repos = sorted(by_repo)
    rng.shuffle(repos)

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

    selected = selected[:n]
    payload = {
        "seed": seed,
        "n": len(selected),
        "instance_ids": selected,
        "repo_counts": dict(Counter(repo_from_instance_id(i) for i in selected)),
    }
    if output_path is not None:
        save_json(output_path, payload)
    return selected


def filter_leakage(
    rows: list[dict[str, Any]],
    *,
    verified_ids: set[str],
    id_key: str = "instance_id",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    leaked = 0
    for row in rows:
        iid = row.get(id_key) or row.get("id")
        if iid in verified_ids:
            leaked += 1
            continue
        kept.append(row)
    stats = {"input": len(rows), "kept": len(kept), "leaked_verified": leaked}
    return kept, stats


def normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coerce tool_calls into OpenAI shape with JSON-string arguments for HF templates."""
    out: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        normalized = dict(tc)
        fn = normalized.get("function")
        if isinstance(fn, dict):
            fn = dict(fn)
            args = fn.get("arguments")
            if isinstance(args, dict):
                fn["arguments"] = json.dumps(args, ensure_ascii=False)
            normalized["function"] = fn
        normalized.setdefault("type", "function")
        out.append(normalized)
    return out


def looks_like_native_tool_format(messages: list[dict[str, Any]]) -> bool:
    """True when the trajectory uses assistant tool_calls (not legacy THOUGHT/ACTION text)."""
    saw_tool_calls = False
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "assistant":
            if msg.get("tool_calls"):
                saw_tool_calls = True
            elif "THOUGHT:" in content and "ACTION:" in content:
                return False
        if role == "user" and "<returncode>" in content and not saw_tool_calls:
            # Legacy mini-swe text-action turns surface shell output as user messages.
            return False
    return saw_tool_calls


def attach_tool_call_ids(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair tool responses with assistant tool_call ids for Gemma4 chat templates."""
    out: list[dict[str, Any]] = []
    pending_ids: list[str] = []
    for msg in messages:
        normalized = dict(msg)
        normalized["content"] = normalized.get("content") or ""
        if normalized.get("role") == "assistant" and normalized.get("tool_calls"):
            pending_ids = [
                tool_call["id"]
                for tool_call in normalized["tool_calls"]
                if isinstance(tool_call, dict) and tool_call.get("id")
            ]
        if normalized.get("role") == "tool" and not normalized.get("tool_call_id") and pending_ids:
            normalized["tool_call_id"] = pending_ids.pop(0)
        out.append(normalized)
    return out


def agent_messages_to_gemma(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize SWE-agent messages into Gemma chat format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            content = "".join(text_parts)

        normalized: dict[str, Any] = {"role": role, "content": content or ""}
        if msg.get("tool_calls"):
            normalized["tool_calls"] = normalize_tool_calls(msg["tool_calls"])
        if role == "tool" and msg.get("name"):
            normalized["name"] = msg["name"]
        if role == "tool" and msg.get("tool_call_id"):
            normalized["tool_call_id"] = msg["tool_call_id"]
        out.append(normalized)
    return attach_tool_call_ids(out)


def prepare_gemma4_chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare stored SFT messages for Gemma4 `apply_chat_template`."""
    return attach_tool_call_ids(agent_messages_to_gemma(messages))


def trajectory_to_sft_row(
    *,
    instance_id: str,
    messages: list[dict[str, Any]],
    source: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "instance_id": instance_id,
        "messages": agent_messages_to_gemma(messages),
        "source": source,
        "metadata": metadata or {},
    }


def estimate_message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(x)) for x in content)
        if msg.get("tool_calls"):
            total += len(json.dumps(msg["tool_calls"], ensure_ascii=False))
    return total


def estimate_message_tokens(messages: list[dict[str, Any]], *, chars_per_token: float = 3.2) -> int:
    """Fast token-length proxy for prep-time filtering without loading a tokenizer."""
    return int(estimate_message_chars(messages) / chars_per_token)


def count_chat_tokens(messages: list[dict[str, Any]], tokenizer) -> int:
    prepared = prepare_gemma4_chat_messages(messages)
    text = tokenizer.apply_chat_template(prepared, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def truncate_token_ids(input_ids: list[int], max_tokens: int, *, mode: str = "keep_end") -> list[int]:
    if len(input_ids) <= max_tokens:
        return input_ids
    if mode == "keep_end":
        return input_ids[-max_tokens:]
    return input_ids[:max_tokens]


def ensure_messages_fit_token_budget(
    messages: list[dict[str, Any]],
    tokenizer,
    max_tokens: int,
    *,
    min_messages: int = 1,
) -> tuple[list[dict[str, Any]], bool]:
    """Drop tail messages until the chat template fits max_tokens."""
    trimmed = list(messages)
    changed = False
    while len(trimmed) > min_messages and count_chat_tokens(trimmed, tokenizer) > max_tokens:
        trimmed.pop()
        changed = True
    return trimmed, changed


def truncate_agent_messages_smart_tail(
    messages: list[dict[str, Any]],
    tokenizer,
    max_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep system + first user, then the longest suffix of turns that fits."""
    original_tokens = count_chat_tokens(messages, tokenizer)
    meta: dict[str, Any] = {
        "truncated": False,
        "policy": "smart_tail",
        "original_tokens": original_tokens,
        "final_tokens": original_tokens,
        "dropped_messages": 0,
    }
    if original_tokens <= max_tokens:
        return messages, meta

    head_len = min(2, len(messages))
    head = messages[:head_len]
    tail = messages[head_len:]
    kept = 0
    for n in range(len(tail), 0, -1):
        candidate = head + tail[-n:]
        token_count = count_chat_tokens(candidate, tokenizer)
        if token_count <= max_tokens:
            kept = n
            meta["final_tokens"] = token_count
            break

    if kept:
        meta["truncated"] = True
        meta["dropped_messages"] = len(tail) - kept
        return head + tail[-kept:], meta

    meta["truncated"] = True
    meta["dropped_messages"] = len(tail)
    meta["fallback"] = "token_keep_end"
    return head, meta


def chunk_agent_messages_for_sft(
    messages: list[dict[str, Any]],
    tokenizer,
    max_tokens: int,
) -> list[tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Split a long trajectory into contiguous chunks that each fit max_tokens."""
    original_tokens = count_chat_tokens(messages, tokenizer)
    if original_tokens <= max_tokens:
        return [
            (
                messages,
                {
                    "truncated": False,
                    "policy": "chunk",
                    "chunk_index": 0,
                    "num_chunks": 1,
                    "original_tokens": original_tokens,
                    "final_tokens": original_tokens,
                },
            )
        ]

    head_len = min(2, len(messages))
    head = messages[:head_len]
    tail = messages[head_len:]
    chunks: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    start = 0
    while start < len(tail):
        end = start
        last_fit = start
        while end < len(tail):
            candidate = head + tail[start : end + 1]
            if count_chat_tokens(candidate, tokenizer) <= max_tokens:
                last_fit = end + 1
                end += 1
            else:
                break
        if last_fit == start:
            last_fit = start + 1
        chunk_messages = head + tail[start:last_fit]
        chunk_messages, _ = ensure_messages_fit_token_budget(
            chunk_messages,
            tokenizer,
            max_tokens,
            min_messages=head_len,
        )
        chunks.append(
            (
                chunk_messages,
                {
                    "truncated": True,
                    "policy": "chunk",
                    "original_tokens": original_tokens,
                    "final_tokens": count_chat_tokens(chunk_messages, tokenizer),
                    "chunk_start": start,
                    "chunk_end": last_fit,
                },
            )
        )
        start = last_fit

    for idx, (chunk_messages, meta) in enumerate(chunks):
        meta["chunk_index"] = idx
        meta["num_chunks"] = len(chunks)
        meta["final_tokens"] = count_chat_tokens(chunk_messages, tokenizer)
    return chunks


def expand_sft_rows_for_context_limit(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    max_tokens: int,
    policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply smart truncation or turn-chunking before tokenization."""
    if policy == "none":
        return rows, {"policy": "none", "input_rows": len(rows), "output_rows": len(rows)}

    expanded: list[dict[str, Any]] = []
    truncated_rows = 0
    chunked_rows = 0
    total_chunks = 0

    for row in rows:
        messages = row["messages"]
        if policy == "smart_tail":
            new_messages, meta = truncate_agent_messages_smart_tail(messages, tokenizer, max_tokens)
            out = dict(row)
            out["messages"] = new_messages
            out.setdefault("metadata", {})
            out["metadata"] = {**out.get("metadata", {}), "context_limit": meta}
            expanded.append(out)
            if meta.get("truncated"):
                truncated_rows += 1
            continue

        if policy == "chunk":
            chunks = chunk_agent_messages_for_sft(messages, tokenizer, max_tokens)
            if len(chunks) > 1:
                chunked_rows += 1
            total_chunks += len(chunks)
            for chunk_messages, meta in chunks:
                out = dict(row)
                out["messages"] = chunk_messages
                out.setdefault("metadata", {})
                out["metadata"] = {**out.get("metadata", {}), "context_limit": meta}
                if meta.get("num_chunks", 1) > 1:
                    suffix = f"chunk{meta['chunk_index']}"
                    out["instance_id"] = f"{row['instance_id']}::{suffix}"
                expanded.append(out)
            continue

        raise ValueError(f"Unsupported truncation policy: {policy}")

    stats = {
        "policy": policy,
        "max_tokens": max_tokens,
        "input_rows": len(rows),
        "output_rows": len(expanded),
        "truncated_rows": truncated_rows,
        "chunked_rows": chunked_rows,
        "total_chunks": total_chunks if policy == "chunk" else len(rows),
    }
    return expanded, stats
