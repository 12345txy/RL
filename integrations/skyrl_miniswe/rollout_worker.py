"""Rollout scheduling: pull queue (default) or legacy Ray remote."""

from __future__ import annotations

import os
from typing import Any

import ray

from integrations.skyrl_miniswe.rollout_core import init_and_run as _init_and_run_local
from integrations.skyrl_miniswe.rollout_queue import run_rollout_job, use_pull_rollout
from skyrl.train.generators.base import TrajectoryID, TrainingPhase

DOCKER_RAY_RESOURCE = os.environ.get("SKYRL_DOCKER_RAY_RESOURCE", "docker_node")


def _docker_remote_options() -> dict[str, Any]:
    if os.environ.get("SKYRL_REQUIRE_DOCKER_NODE", "0") != "0":
        return {"num_cpus": 0.01, "resources": {DOCKER_RAY_RESOURCE: 0.01}}
    return {"num_cpus": 0.01}


@ray.remote(num_cpus=0.01)
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
):
    return _init_and_run_local(
        instance,
        litellm_model_name,
        sweagent_config,
        miniswe_traj_dir,
        data_source,
        sampling_params,
        trajectory_id,
        global_step,
        training_phase,
    )


async def schedule_init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    miniswe_traj_dir: str,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
    if use_pull_rollout():
        return await run_rollout_job(
            instance,
            litellm_model_name,
            sweagent_config,
            miniswe_traj_dir,
            data_source,
            sampling_params,
            trajectory_id,
            global_step,
            training_phase,
        )
    return await init_and_run.options(**_docker_remote_options()).remote(
        instance,
        litellm_model_name,
        sweagent_config,
        miniswe_traj_dir,
        data_source,
        sampling_params,
        trajectory_id,
        global_step,
        training_phase,
    )
