"""SkyRL GRPO entrypoint for Gemma4-12B + Mini-SWE-Agent SWE-RL."""

from __future__ import annotations

import sys

import ray
from skyrl.train.config import SkyRLGymConfig, make_config
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray

from integrations.skyrl_miniswe.generator import MiniSWEGeneratorConfig, MiniSweAgentGenerator

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


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg):
    exp = MiniSWEPPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = MiniSWEConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
