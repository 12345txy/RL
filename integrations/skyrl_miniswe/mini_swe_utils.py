"""Docker/Podman helpers for Mini-SWE-Agent SWE-bench evaluation."""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from jinja2 import Template
from loguru import logger
from minisweagent.environments import Environment, get_environment


class MiniSWEEvaluationResult(TypedDict):
    instance_id: str
    resolved: bool
    eval_error: Optional[str]


def get_sb_environment(config: dict, instance: dict, data_source: str) -> Environment:
    env_config = config.setdefault("environment", {})
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_docker_image_name(instance, data_source)
    if env_config["environment_class"] == "docker":
        env_config["image"] = image_name
    elif env_config["environment_class"] == "singularity":
        env_config["image"] = f"docker://{image_name}"
    env = get_environment(env_config)
    if startup_command := config.get("run", {}).get("env_startup_command"):
        startup_command = Template(startup_command).render(**instance)
        out = env.execute(startup_command)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def get_docker_image_name(instance: dict, data_source: str) -> str:
    image_name = instance.get("image_name")
    if image_name is None:
        iid = instance["instance_id"]
        if "swe-gym" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_s_")
            image_name = f"docker.io/xingyaoww/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        elif "swe-bench" in data_source.lower():
            id_docker_compatible = iid.replace("__", "_1776_")
            image_name = f"docker.io/swebench/sweb.eval.x86_64.{id_docker_compatible}:latest".lower()
        else:
            raise NotImplementedError(f"Data source: {data_source} is not supported")
    return image_name


def evaluate_trajectory(
    instance: Dict[str, Any],
    model_patch: str,
    sweagent_config: dict,
    data_source: str,
) -> MiniSWEEvaluationResult:
    ret = MiniSWEEvaluationResult(instance_id=instance["instance_id"], resolved=False, eval_error=None)
    env = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
    except Exception as e:
        ret["eval_error"] = f"Env creation failed with {e}"
        logger.info(f"Starting environment failed with exception: {e}\n{traceback.format_exc()}")
        return ret

    delimiter = f"PATCH_{uuid.uuid4().hex}"
    command = f"git apply <<'{delimiter}'\n{model_patch}\n{delimiter}"
    obs = env.execute(command)

    if obs["returncode"] != 0:
        ret["eval_error"] = obs["output"]
    else:
        eval_script = instance["eval_script"]
        eval_cmd = f"bash <<'EOF'\n{eval_script}\nEOF"
        obs = env.execute(eval_cmd, timeout=3600)
        ret["resolved"] = obs["returncode"] == 0
        ret["eval_error"] = (
            f"(truncated to last 1000 characters)\n{obs['output'][-1000:]}" if not ret["resolved"] else None
        )
    return ret


def save_traj(
    agent: Any,
    path: Path | str | None,
    *,
    exit_status: str | None = None,
    result: Any = None,
    extra_info: dict | None = None,
    reward: int | None = None,
    eval_error: str | None = None,
) -> dict:
    """Compat wrapper for mini-swe-agent v2 (agent.save replaces run.utils.save)."""
    extra: dict[str, Any] = {}
    if exit_status is not None:
        extra["exit_status"] = exit_status
    if result is not None:
        extra["result"] = result
    if extra_info is not None:
        extra["extra_info"] = extra_info
    if reward is not None:
        extra["reward"] = reward
    if eval_error is not None:
        extra["eval_error"] = eval_error
    out_path = Path(path) if path is not None else None
    return agent.save(out_path, extra)
