"""SkyRL GeneratorInterface for Mini-SWE-Agent + real Docker SWE-bench rollouts."""

from __future__ import annotations

import asyncio
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray
import yaml
from loguru import logger
from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_path
from minisweagent.models import get_model
from integrations.skyrl_miniswe.mini_swe_utils import evaluate_trajectory, get_sb_environment, save_traj
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import BatchMetadata, GeneratorInput, GeneratorOutput, TrajectoryID, TrainingPhase
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.utils import get_response_ids_and_loss_mask_from_messages, get_rollout_metrics

DOCKER_RAY_RESOURCE = os.environ.get("SKYRL_DOCKER_RAY_RESOURCE", "docker_node")


@dataclass
class MiniSWEGeneratorConfig(GeneratorConfig):
    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""


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
    """Schedule Docker rollouts on CPU Ray workers tagged with docker_node."""
    if os.environ.get("SKYRL_REQUIRE_DOCKER_NODE", "0") == "1":
        return {"num_cpus": 0.01, "resources": {DOCKER_RAY_RESOURCE: 0.01}}
    return {"num_cpus": 0.01}


@ray.remote(**_docker_remote_options())
def init_and_run(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    generator_cfg: GeneratorConfig,
    data_source: str,
    sampling_params: dict,
    trajectory_id: TrajectoryID,
    global_step: int,
    training_phase: TrainingPhase,
):
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
        traj_root = Path(generator_cfg.miniswe_traj_dir) / f"step_{global_step}" / training_phase
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


class MiniSweAgentGenerator(SkyRLGymGenerator):
    def __init__(
        self,
        generator_cfg: GeneratorConfig,
        skyrl_gym_cfg: SkyRLGymConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        super().__init__(generator_cfg, skyrl_gym_cfg, inference_engine_client, tokenizer)
        host = generator_cfg.inference_engine.http_endpoint_host
        port = generator_cfg.inference_engine.http_endpoint_port
        self.base_url = f"http://{host}:{port}"
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.litellm_model_name = "openai/" + self.model_name
        if self.generator_cfg.chat_template.name_or_path is not None:
            raise NotImplementedError("MiniSWEAgentGenerator doesn't support custom chat template")

    async def minisweagent_agent_loop(
        self,
        prompt: ConversationType,
        env_extras: dict[str, Any],
        max_tokens: int,
        max_input_length: int,
        sampling_params: dict[str, Any],
        trajectory_id: TrajectoryID,
        batch_metadata: BatchMetadata,
    ):
        del prompt
        sweagent_config = yaml.safe_load(get_config_path(self.generator_cfg.miniswe_config_path).read_text())
        messages, reward, error = await init_and_run.remote(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        del error
        if not messages:
            return None, None, None, None, None, None

        response_messages = messages[2:]
        for message in messages[:2]:
            assert message["role"] in ("system", "user")

        initial_input_ids = self.tokenizer.apply_chat_template(
            messages[:2], add_generation_prompt=False, return_dict=False, tokenize=True
        )
        initial_prompt_length = len(initial_input_ids)

        last_idx = len(response_messages) - 1
        while response_messages[last_idx]["role"] == "user":
            last_idx -= 1
        if last_idx < 0:
            raise ValueError(
                "Found no assistant messages. Ensure Mini-SWE-Agent can reach the SkyRL vLLM HTTP endpoint."
            )
        response_messages = response_messages[: last_idx + 1]

        response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
            response_messages,
            self.tokenizer,
            assistant_logprobs=None,
        )
        prompt_ids = initial_input_ids
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length
        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        return (response_ids, reward, stop_reason, loss_mask, prompt_ids, None)

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        env_extras = input_batch["env_extras"]
        trajectory_ids = input_batch["trajectory_ids"]
        batch_metadata = input_batch["batch_metadata"]
        max_tokens = self.generator_cfg.sampling_params.max_generate_length
        max_input_length = self.generator_cfg.max_input_length
        sampling_params = get_sampling_params_for_backend(
            self.generator_cfg.inference_engine.backend,
            self.generator_cfg.sampling_params,
        )

        tasks = [
            self.minisweagent_agent_loop(
                prompts[i],
                env_extras[i],
                max_tokens=max_tokens,
                max_input_length=max_input_length,
                sampling_params=sampling_params,
                trajectory_id=trajectory_ids[i],
                batch_metadata=batch_metadata,
            )
            for i in range(len(prompts))
        ]
        all_outputs = await asyncio.gather(*tasks)

        responses = [output[0] for output in all_outputs if output[0] is not None]
        rewards = [output[1] for output in all_outputs if output[0] is not None]
        stop_reasons = [output[2] for output in all_outputs if output[0] is not None]
        loss_masks = [output[3] for output in all_outputs if output[0] is not None]
        prompt_token_ids = [output[4] for output in all_outputs if output[0] is not None]
        if not responses:
            raise ValueError(
                "No valid trajectories in this batch. Check Docker on CPU Ray workers and vLLM HTTP endpoint."
            )

        rollout_metrics = get_rollout_metrics(responses, rewards)
        return {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }
