"""mini-swe-agent multi-turn rollout generator for Sky-RL-style training."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MINISWE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Execute bash in /testbed",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Edit files in /testbed",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["command", "path"],
            },
        },
    },
]


@dataclass
class RolloutResult:
    instance_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    patch: str = ""
    reward: float = 0.0
    num_steps: int = 0


class MiniSweRolloutGenerator:
    """Lightweight rollout loop compatible with Sky-RL GeneratorInterface patterns."""

    def __init__(
        self,
        *,
        api_base: str = "http://127.0.0.1:8000/v1",
        model: str = "gemma-4-12B-it",
        max_turns: int = 50,
        temperature: float = 0.7,
        step_penalty: float = 0.01,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.max_turns = max_turns
        self.temperature = temperature
        self.step_penalty = step_penalty

    def _chat(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": MINISWE_TOOLS,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": 2048,
        }
        req = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def rollout(self, instance: dict[str, Any]) -> RolloutResult:
        from integrations.skyrl_miniswe.reward import compute_resolve_reward

        iid = instance["instance_id"]
        problem = instance.get("problem_statement") or instance.get("issue_body") or ""
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that can interact with a computer shell "
                    "to solve programming tasks. Fix the issue in /testbed and return a git patch."
                ),
            },
            {"role": "user", "content": problem},
        ]
        patch = ""
        steps = 0
        for _ in range(self.max_turns):
            steps += 1
            resp = self._chat(messages)
            msg = resp["choices"][0]["message"]
            messages.append(msg)
            tool_calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()

            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except json.JSONDecodeError:
                        args = {}
                    if name == "execute_bash":
                        obs = f"$ {args.get('command','')}\n(exit 0)"
                    else:
                        obs = f"[{name}] ok"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", "call"),
                            "name": name,
                            "content": obs,
                        }
                    )
            elif "diff --git" in content:
                patch = content
                break
            elif content and steps >= self.max_turns:
                patch = content

        reward = compute_resolve_reward(
            instance_id=iid,
            patch=patch,
            step_penalty=self.step_penalty,
            num_steps=steps,
        )
        return RolloutResult(instance_id=iid, messages=messages, patch=patch, reward=reward, num_steps=steps)

    def generate_batch(self, instances: list[dict[str, Any]]) -> list[RolloutResult]:
        return [self.rollout(inst) for inst in instances]


def load_rl_pool(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
