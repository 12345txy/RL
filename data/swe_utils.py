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
            normalized["tool_calls"] = msg["tool_calls"]
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
