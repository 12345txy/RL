"""MBPP prompt helpers aligned with lm-evaluation-harness mbpp_instruct task."""


from __future__ import annotations


def build_lmeval_mbpp_prompt(text: str, test_list: list[str]) -> str:
    """Match lm_eval/tasks/mbpp/mbpp_instruct.yaml doc_to_text (chat user content)."""
    tests = "\n".join(test_list[:3])
    return (
        "You are an expert Python programmer, and here is your task:\n"
        f"{text.strip()}\n"
        "Your code should pass these tests:\n"
        f"{tests}\n"
        "Put your complete solution in a single ```python ... ``` code block."
    )


def normalize_test_imports(test_setup_code: str | list[str] | None) -> list[str]:
    if not test_setup_code:
        return []
    if isinstance(test_setup_code, list):
        return [line for line in test_setup_code if line.strip()]
    return [line for line in test_setup_code.splitlines() if line.strip()]
