"""Code execution rewards for GRPO coding RL."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

CODE_BLOCK_PATTERN = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
BANNED_PATTERNS = (
    "os.system",
    "subprocess",
    "shutil.rmtree",
    "__import__",
    "eval(",
    "exec(",
    "open(",
    "compile(",
)

MAX_WORKERS = min(32, (os.cpu_count() or 8) * 2)
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def _clean_codeblock(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?\s*", "", code, flags=re.IGNORECASE)
        code = re.sub(r"\s*```$", "", code).strip()
    return code


def extract_code(completion: str) -> str | None:
    blocks = [_clean_codeblock(block) for block in CODE_BLOCK_PATTERN.findall(completion)]
    if blocks:
        for code in reversed(blocks):
            if "def " in code:
                return code or None
        code = blocks[-1]
    else:
        code = completion.strip()
    if not code:
        return None
    return _clean_codeblock(code)


def is_safe_code(code: str) -> bool:
    lowered = code.lower()
    return not any(token in lowered for token in BANNED_PATTERNS)


def _build_test_harness(import_lines: str, code: str, test_list: list[str]) -> str:
    imports_block = f"{import_lines}\n\n" if import_lines else ""
    assertions = "\n".join(f"    {line!r}," for line in test_list)
    return f"""{imports_block}{code}

__tests__ = [
{assertions}
]
__passed__ = 0
for __t__ in __tests__:
    try:
        exec(compile(__t__, "<test>", "exec"), globals())
        __passed__ += 1
    except Exception:
        pass
print(__passed__, len(__tests__))
"""


def run_tests(code: str, test_imports: list[str], test_list: list[str], timeout: int = 5) -> tuple[int, int]:
    if not code or not test_list:
        return 0, max(len(test_list), 1)

    import_lines = "\n".join(test_imports) if test_imports else ""
    script = _build_test_harness(import_lines, code, test_list)
    total = len(test_list)
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=max(timeout, total * 2),
        )
        if result.returncode != 0:
            return 0, total
        passed_str, total_str = result.stdout.strip().split()
        return int(passed_str), int(total_str)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return 0, total


@lru_cache(maxsize=4096)
def _cached_run_tests(code: str, import_lines: str, tests_key: tuple[str, ...], timeout: int) -> tuple[int, int]:
    return run_tests(code, list(import_lines.split("\n")) if import_lines else [], list(tests_key), timeout)


def _completion_text(completion) -> str:
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict):
            return completion[0].get("content", "")
        return str(completion)
    return str(completion)


def format_reward(completions, **kwargs) -> list[float]:
    rewards = []
    for completion in completions:
        text = _completion_text(completion)
        code = extract_code(text)
        if code is None or not is_safe_code(code):
            rewards.append(0.0)
            continue
        try:
            ast.parse(code)
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    return rewards


def execution_reward(completions, test_list, test_imports=None, **kwargs) -> list[float]:
    if test_imports is None:
        test_imports = [[] for _ in completions]
    if test_imports and isinstance(test_imports[0], str):
        test_imports = [[imp] if imp else [] for imp in test_imports]

    rewards = [0.0] * len(completions)
    futures = {}

    for idx, completion in enumerate(completions):
        text = _completion_text(completion)
        code = extract_code(text)
        imports = test_imports[idx] if idx < len(test_imports) else []
        tests = test_list[idx] if idx < len(test_list) else []

        if code is None or not is_safe_code(code) or not tests:
            continue

        import_lines = "\n".join(imports) if imports else ""
        tests_key = tuple(tests)
        futures[
            EXECUTOR.submit(_cached_run_tests, code, import_lines, tests_key, 5)
        ] = idx

    for future in as_completed(futures):
        idx = futures[future]
        try:
            passed, total = future.result()
            rewards[idx] = passed / total if total else 0.0
        except Exception:
            rewards[idx] = 0.0

    return rewards
