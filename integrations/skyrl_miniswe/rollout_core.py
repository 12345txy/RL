"""Core Mini-SWE-Agent rollout logic (local Docker, no Ray)."""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model

from integrations.skyrl_miniswe.mini_swe_utils import evaluate_trajectory, get_sb_environment, save_traj
from integrations.skyrl_miniswe.types import TrajectoryID, TrainingPhase


def resolve_docker_executable(sweagent_config: dict) -> None:
    """Ensure mini-swe-agent uses an absolute docker path (workers often lack PATH)."""
    env_cfg = sweagent_config.setdefault("environment", {})
    current = env_cfg.get("executable", "docker")
    if current not in ("docker", "podman"):
        return
    for candidate in (
        os.environ.get("MSWEA_DOCKER_EXECUTABLE"),
        os.environ.get("SKYRL_DOCKER_EXECUTABLE"),
        os.environ.get("DOCKER_EXECUTABLE"),
        shutil.which(current),
        "/usr/bin/docker",
        "/usr/local/bin/docker",
    ):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            env_cfg["executable"] = candidate
            os.environ["MSWEA_DOCKER_EXECUTABLE"] = candidate
            return


def _model_config(sweagent_config: dict) -> dict:
    model_cfg = sweagent_config.get("model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}
        sweagent_config["model"] = model_cfg
    model_cfg.setdefault("cost_tracking", os.environ.get("MSWEA_COST_TRACKING", "ignore_errors"))
    model_cfg.setdefault("model_kwargs", {})
    return model_cfg


def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    miniswe_traj_dir: str,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
) -> tuple[list, int, str | None]:
    from loguru import logger

    resolve_docker_executable(sweagent_config)
    model_config = _model_config(sweagent_config)
    if isinstance(sampling_params, dict):
        model_config["model_kwargs"].update(sampling_params)
    model = get_model(litellm_model_name, model_config)

    agent = None
    extra_info = None
    eval_result = None
    reward = 0
    error = None
    exit_status = "Unknown"
    model_patch = ""
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgent(model, env, **sweagent_config.get("agent", {}))
        run_extra = agent.run(instance["problem_statement"])
        if not isinstance(run_extra, dict):
            raise TypeError(f"agent.run() returned {type(run_extra).__name__}, expected dict")
        exit_status = str(run_extra.get("exit_status", "Unknown"))
        model_patch = str(run_extra.get("submission", "") or "")
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status = type(e).__name__
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        traj_root = Path(miniswe_traj_dir) / f"step_{global_step}" / training_phase
        traj_root.mkdir(parents=True, exist_ok=True)
        filename = f"{instance['instance_id']}_{trajectory_id.repetition_id}.json"
        path = traj_root / filename
        if agent is not None:
            eval_error = None
            if model_patch.strip():
                try:
                    eval_result = evaluate_trajectory(instance, model_patch, sweagent_config, data_source)
                    reward = int(eval_result["resolved"])
                    eval_error = eval_result["eval_error"]
                    if eval_error:
                        error = eval_error
                except Exception as e:
                    eval_error = str(e)
                    error = str(e)
            save_traj(
                agent,
                path,
                exit_status=exit_status,
                result=eval_result if eval_result is not None else model_patch,
                extra_info=extra_info,
                reward=reward,
                eval_error=eval_error,
            )

    return (agent.messages if agent is not None else [], reward, error)
