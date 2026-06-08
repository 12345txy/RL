#!/usr/bin/env python3
"""Print pass@1 metrics from lm-eval result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def find_latest_results(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.rglob("results_*.json"))
    return candidates[-1] if candidates else None


def extract_pass_at_1(results: dict) -> list[dict]:
    metrics = []
    for task, values in results.get("results", {}).items():
        pass_key = next((k for k in values if k.startswith("pass@1,")), None)
        if pass_key is None:
            pass_key = next((k for k in values if k.startswith("pass@1")), None)
        if pass_key is None:
            pass_key = next((k for k in values if k.startswith("pass_at_1,")), None)
        if pass_key is None:
            pass_key = next((k for k in values if k.startswith("pass_at_1")), None)
        if pass_key is None:
            continue
        stderr_key = pass_key.replace("pass@1", "pass@1_stderr").replace("pass_at_1", "pass_at_1_stderr")
        metrics.append(
            {
                "task": task,
                "pass@1": values[pass_key],
                "stderr": values.get(stderr_key),
            }
        )
    return metrics


def find_sample_files(results_file: Path) -> list[Path]:
    ts = results_file.stem.removeprefix("results_")
    matched = sorted(results_file.parent.glob(f"samples_*_{ts}.jsonl"))
    if matched:
        return matched
    return sorted(results_file.parent.glob("samples_*.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results/lm_eval")
    parser.add_argument("--output", type=str, default=None, help="Optional summary JSON path")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_file = find_latest_results(results_dir)
    if results_file is None:
        raise FileNotFoundError(f"No lm-eval results_*.json under {results_dir}")

    payload = json.loads(results_file.read_text(encoding="utf-8"))
    sample_files = find_sample_files(results_file)
    summary = {
        "results_file": str(results_file),
        "sample_files": [str(p) for p in sample_files],
        "metrics": extract_pass_at_1(payload),
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Results file: {results_file}")
    for sample_file in sample_files:
        print(f"Samples file: {sample_file}")
    for metric in summary["metrics"]:
        stderr = metric["stderr"]
        stderr_text = f" ± {stderr:.4f}" if isinstance(stderr, (int, float)) else ""
        print(f"{metric['task']:22s} pass@1={metric['pass@1']:.4f}{stderr_text}")


if __name__ == "__main__":
    main()
