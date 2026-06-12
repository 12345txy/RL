"""SWE-Gym binary resolve reward for Sky-RL / custom RL loops."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def apply_patch_in_repo(repo_dir: Path, patch: str) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"
    patch_file = repo_dir / "_model.patch"
    patch_file.write_text(patch, encoding="utf-8")
    proc = subprocess.run(
        ["git", "apply", "--check", str(patch_file)],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, proc.stderr or proc.stdout
    proc = subprocess.run(
        ["git", "apply", str(patch_file)],
        cwd=repo_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, proc.stderr or proc.stdout
    return True, "applied"


def run_swebench_eval_single(
    *,
    instance_id: str,
    patch: str,
    predictions_path: Path,
    run_id: str,
    max_workers: int = 1,
) -> bool:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        instance_id: {
            "model_name_or_path": "policy",
            "instance_id": instance_id,
            "model_patch": patch,
        }
    }
    predictions_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    proc = subprocess.run(
        [
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            "princeton-nlp/SWE-bench_Verified",
            "--split",
            "test",
            "--predictions_path",
            str(predictions_path),
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
            "--instance_ids",
            instance_id,
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False

    report_dir = Path("logs/run_evaluation") / run_id / instance_id
    report = report_dir / "report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        return bool(data.get("resolved"))
    return False


def compute_resolve_reward(
    *,
    instance_id: str,
    patch: str,
    step_penalty: float = 0.01,
    num_steps: int = 0,
    use_harness: bool = False,
) -> float:
    """Return 1.0 if resolved else 0.0, minus optional step penalty."""
    if not patch.strip():
        return 0.0

    if use_harness:
        with tempfile.TemporaryDirectory() as tmp:
            preds = Path(tmp) / "preds.json"
            resolved = run_swebench_eval_single(
                instance_id=instance_id,
                patch=patch,
                predictions_path=preds,
                run_id=f"rl-reward-{instance_id}",
            )
    else:
        resolved = patch.strip().startswith("diff --git") and len(patch) > 40

    reward = 1.0 if resolved else 0.0
    reward -= step_penalty * num_steps
    return max(reward, 0.0)


def batch_rewards(
    instances: list[dict[str, Any]],
    patches: list[str],
    *,
    step_counts: list[int] | None = None,
    step_penalty: float = 0.01,
) -> list[float]:
    step_counts = step_counts or [0] * len(instances)
    return [
        compute_resolve_reward(
            instance_id=inst["instance_id"],
            patch=patch,
            step_penalty=step_penalty,
            num_steps=steps,
        )
        for inst, patch, steps in zip(instances, patches, step_counts)
    ]
