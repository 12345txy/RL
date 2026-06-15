"""Ray remote worker for Mini-SWE-Agent Docker rollouts (minimal imports for pickling)."""

from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path
from typing import Any

import ray
from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model

from integrations.skyrl_miniswe.mini_swe_utils import evaluate_trajectory, get_sb_environment, save_traj
from skyrl.train.generators.base import TrajectoryID, TrainingPhase

DOCKER_RAY_RESOURCE = os.environ.get("SKYRL_DOCKER_RAY_RESOURCE", "docker_node")


class DefaultAgentWithReminder(DefaultAgent):
    def get_observation(self, response: dict) -> dict:
        output = self.execute_action(self.parse_action(response))
        observation = self.render_template(self.config.action_observation_template, output=output)
        remaining = self.config.step_limit - self.model.n_calls
        if remaining == 1:
            observation = (
                f"{observation}\nREMINDER: You only have 1 turn left. Please provide the final answer"
            )
        elif remaining > 1:
            observation = f"{observation}\nREMINDER: You have {remaining} turns left to arrive at the solution."
        self.add_message("user", observation)
        return output


def _docker_remote_options() -> dict[str, Any]:
    # Default on: skyrl_entrypoint runs on GPU workers that do not inherit the driver's shell env.
    if os.environ.get("SKYRL_REQUIRE_DOCKER_NODE", "1") != "0":
        return {"num_cpus": 0.01, "resources": {DOCKER_RAY_RESOURCE: 0.01}}
    return {"num_cpus": 0.01}


def _resolve_docker_executable(sweagent_config: dict) -> None:
    """Ensure mini-swe-agent uses an absolute docker path (Ray workers often lack PATH)."""
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
    from loguru import logger

    _resolve_docker_executable(sweagent_config)
    model_config = sweagent_config.get("model", {})
    model_config.setdefault("model_kwargs", {}).update(sampling_params)
    model = get_model(litellm_model_name, model_config)

    agent = None
    extra_info = None
    result = None
    reward = 0
    error = None
    try:
        env = get_sb_environment(sweagent_config, instance, data_source)
        agent = DefaultAgentWithReminder(model, env, **sweagent_config.get("agent", {}))
        exit_status, result = agent.run(instance["problem_statement"])
    except Exception as e:
        logger.error(f"Error processing instance {instance['instance_id']}: {e}", exc_info=True)
        exit_status, result = type(e).__name__, str(e)
        error = str(e)
        extra_info = {"traceback": traceback.format_exc()}
    finally:
        traj_root = Path(miniswe_traj_dir) / f"step_{global_step}" / training_phase
        traj_root.mkdir(parents=True, exist_ok=True)
        filename = f"{instance['instance_id']}_{trajectory_id.repetition_id}.json"
        path = traj_root / filename
        if agent is not None:
            eval_error = None
            try:
                result = evaluate_trajectory(instance, result, sweagent_config, data_source)
                reward = int(result["resolved"])
                eval_error = result["eval_error"]
                if eval_error:
                    error = eval_error
            except Exception as e:
                eval_error = str(e)
                error = str(e)
            save_traj(
                agent,
                path,
                exit_status=exit_status,
                result=result,
                extra_info=extra_info,
                reward=reward,
                eval_error=eval_error,
            )

    return (agent.messages if agent is not None else [], reward, error)


def schedule_init_and_run(*args, **kwargs):
    """Apply docker_node scheduling options at call time."""
    return init_and_run.options(**_docker_remote_options()).remote(*args, **kwargs)
