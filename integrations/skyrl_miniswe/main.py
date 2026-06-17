"""SkyRL GRPO entrypoint for Gemma4-12B + Mini-SWE-Agent SWE-RL."""

from __future__ import annotations

import os
import sys

import ray
from skyrl.train.config import SkyRLGymConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

from integrations.skyrl_miniswe.cached_prompt_dataset import CachedPromptDataset
from integrations.skyrl_miniswe.generator import MiniSWEGeneratorConfig, MiniSweAgentGenerator
from integrations.skyrl_miniswe.rollout_queue import start_rollout_queue_server, use_pull_rollout

MiniSWEConfig = make_config(generator_cls=MiniSWEGeneratorConfig)


class MiniSWEPPOExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return MiniSweAgentGenerator(
            generator_cfg=cfg.generator,
            skyrl_gym_cfg=SkyRLGymConfig(max_env_workers=0),
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            model_name=self.cfg.trainer.policy.model.path,
        )

    def get_train_dataset(self):
        prompts_dataset = CachedPromptDataset(
            datasets=self.cfg.data.train_data,
            tokenizer=self.tokenizer,
            max_prompt_length=self.cfg.trainer.max_prompt_length,
            num_workers=8,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be at least as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def get_eval_dataset(self):
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            return CachedPromptDataset(
                datasets=self.cfg.data.val_data,
                tokenizer=self.tokenizer,
                max_prompt_length=self.cfg.trainer.max_prompt_length,
                num_workers=8,
            )
        return None


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    queue_server = None
    if use_pull_rollout():
        host = os.environ.get("SKYRL_ROLLOUT_QUEUE_HOST", "127.0.0.1")
        port = int(os.environ.get("SKYRL_ROLLOUT_QUEUE_PORT", "9000"))
        queue_server = start_rollout_queue_server(host=host, port=port)
        print(
            f"==> Pull rollout queue listening on http://{host}:{port} "
            f"(CPU workers: SKYRL_ROLLOUT_QUEUE_URL=http://127.0.0.1:{port})",
            flush=True,
        )
    try:
        exp = MiniSWEPPOExp(cfg)
        exp.run()
    finally:
        if queue_server is not None:
            queue_server.shutdown()


def main() -> None:
    cfg = MiniSWEConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
