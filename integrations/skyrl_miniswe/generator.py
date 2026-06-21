"""SkyRL GeneratorInterface for Mini-SWE-Agent + real Docker SWE-bench rollouts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import yaml
from minisweagent.config import get_config_path
from integrations.skyrl_miniswe.rollout_worker import schedule_init_and_run
from skyrl.backends.skyrl_train.inference_engines.base import ConversationType
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.backends.skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl.train.config import GeneratorConfig, SkyRLGymConfig
from skyrl.train.generators.base import BatchMetadata, GeneratorInput, GeneratorOutput, TrajectoryID
from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator
from skyrl.train.generators.utils import get_rollout_metrics

# mini-swe-agent v2 adds internal `exit` messages; SkyRL only tokenizes user/assistant/tool.
_TRAINABLE_ROLES = frozenset({"user", "assistant", "tool"})
_TOKENIZER_MESSAGE_KEYS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


def _training_messages(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") in _TRAINABLE_ROLES]


def _response_messages_for_training(messages: list[dict]) -> list[dict]:
    """Keep system/user prefix intact; drop mini-swe internal roles from the rollout tail."""
    if len(messages) <= 2:
        return _training_messages(messages)
    head = messages[:2]
    tail = _training_messages(messages[2:])
    while tail and tail[-1]["role"] == "user":
        tail.pop()
    return head + tail


def _sanitize_message_for_tokenizer(message: dict) -> dict:
    clean = {key: message[key] for key in _TOKENIZER_MESSAGE_KEYS if key in message}
    if clean.get("content") is None:
        clean["content"] = ""
    return clean


def _tokenize_chat(tokenizer, messages: list[dict]) -> list[int]:
    sanitized = [_sanitize_message_for_tokenizer(m) for m in messages]
    token_ids = tokenizer.apply_chat_template(
        sanitized,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=False,
    )
    if isinstance(token_ids, dict):
        return list(token_ids["input_ids"])
    return list(token_ids)


def _response_ids_and_loss_mask_from_messages(
    prefix_messages: list[dict],
    response_messages: list[dict],
    tokenizer,
) -> tuple[list[int], list[int]]:
    """Tokenize multi-turn tool-calling trajectories in full conversational context.

    Gemma4 rollouts from vLLM use ``<|tool_call>`` tokens that do not match the
    per-message generation header expected by SkyRL's default helper when messages
    are encoded in isolation. Incremental full-context tokenization avoids that mismatch.
    """
    prompt_ids = _tokenize_chat(tokenizer, prefix_messages)
    prev_ids = prompt_ids
    response_ids: list[int] = []
    loss_mask: list[int] = []

    for idx in range(len(response_messages)):
        message = response_messages[idx]
        curr_ids = _tokenize_chat(tokenizer, prefix_messages + response_messages[: idx + 1])
        message_ids = curr_ids[len(prev_ids) :]
        prev_ids = curr_ids
        response_ids.extend(message_ids)
        train_flag = 1 if message.get("role") == "assistant" else 0
        loss_mask.extend([train_flag] * len(message_ids))

    return response_ids, loss_mask


@dataclass
class MiniSWEGeneratorConfig(GeneratorConfig):
    miniswe_config_path: str = ""
    miniswe_traj_dir: str = ""
    # 0 = no extra cap. Default 12288 via run_rl_skyrl.sh balances VRAM vs truncation.
    max_train_seq_len: int = 0


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
        messages, reward, error = await schedule_init_and_run(
            env_extras["instance"],
            self.litellm_model_name,
            sweagent_config,
            self.generator_cfg.miniswe_traj_dir,
            env_extras["data_source"],
            sampling_params,
            trajectory_id,
            batch_metadata.global_step,
            batch_metadata.training_phase,
        )
        del error
        if not messages:
            return None, None, None, None, None, None

        train_messages = _response_messages_for_training(messages)
        if len(train_messages) < 3:
            raise ValueError(
                "Rollout produced no trainable assistant turns after filtering mini-swe exit messages."
            )

        response_messages = train_messages[2:]
        for message in train_messages[:2]:
            assert message["role"] in ("system", "user")

        initial_input_ids = _tokenize_chat(self.tokenizer, train_messages[:2])
        initial_prompt_length = len(initial_input_ids)

        if not any(m["role"] == "assistant" for m in response_messages):
            raise ValueError(
                "Found no assistant messages. Ensure Mini-SWE-Agent can reach the SkyRL vLLM HTTP endpoint."
            )

        response_ids, loss_mask = _response_ids_and_loss_mask_from_messages(
            train_messages[:2],
            response_messages,
            self.tokenizer,
        )
        prompt_ids = initial_input_ids
        max_response_tokens = max_tokens + max_input_length - initial_prompt_length
        stop_reason = "complete"
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]

        max_train_seq_len = int(getattr(self.generator_cfg, "max_train_seq_len", 0) or 0)
        if max_train_seq_len > 0:
            max_response_for_train = max(0, max_train_seq_len - len(prompt_ids))
            if len(response_ids) > max_response_for_train:
                response_ids = response_ids[:max_response_for_train]
                loss_mask = loss_mask[:max_response_for_train]
                stop_reason = "length"

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

        batch_size = len(prompts)
        max_rollout_retries = int(
            getattr(self.generator_cfg, "max_rollout_retries", 2)
        )
        outputs: list[Any | None] = [None] * batch_size
        pending = list(range(batch_size))

        for attempt in range(max_rollout_retries + 1):
            if not pending:
                break
            attempt_outputs = await asyncio.gather(
                *[
                    self.minisweagent_agent_loop(
                        prompts[i],
                        env_extras[i],
                        max_tokens=max_tokens,
                        max_input_length=max_input_length,
                        sampling_params=sampling_params,
                        trajectory_id=trajectory_ids[i],
                        batch_metadata=batch_metadata,
                    )
                    for i in pending
                ]
            )
            next_pending: list[int] = []
            for idx, output in zip(pending, attempt_outputs):
                if output[0] is not None:
                    outputs[idx] = output
                else:
                    next_pending.append(idx)
            pending = next_pending

        if any(output is None for output in outputs):
            failed = sum(output is None for output in outputs)
            raise ValueError(
                f"{failed}/{batch_size} rollouts failed after {max_rollout_retries + 1} attempts. "
                "Check CPU pull workers (bash scripts/run_rollout_pull_worker.sh), "
                "rollout queue port 9000 SSH tunnel, and vLLM HTTP endpoint."
            )

        responses = [output[0] for output in outputs]
        rewards = [output[1] for output in outputs]
        stop_reasons = [output[2] for output in outputs]
        loss_masks = [output[3] for output in outputs]
        prompt_token_ids = [output[4] for output in outputs]

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
