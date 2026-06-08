#!/usr/bin/env python3
"""GRPO training for coding RL on MBPP."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import load_from_disk
from peft import LoraConfig
from swanlab.integration.transformers import SwanLabCallback
from trl import GRPOConfig, GRPOTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rewards.code_reward import execution_reward, format_reward

DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _gemma4_language_lora_targets(num_layers: int) -> list[str]:
    """Gemma4 wraps some projections in Gemma4ClippableLinear; target LM Linear layers only."""
    targets: list[str] = []
    for layer_idx in range(num_layers):
        prefix = f"model.language_model.layers.{layer_idx}"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            targets.append(f"{prefix}.self_attn.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            targets.append(f"{prefix}.mlp.{name}")
    return targets


def _build_lora_config(model_path: str, *, r: int, alpha: int) -> LoraConfig:
    from transformers import AutoConfig

    target_modules: list[str] = DEFAULT_LORA_TARGETS
    model_name = Path(model_path).name.lower()
    if "gemma" in model_name:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        num_layers = getattr(text_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError(f"Cannot infer num_hidden_layers for Gemma model: {model_path}")
        target_modules = _gemma4_language_lora_targets(num_layers)

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", os.environ.get("ACCELERATE_NUM_PROCESSES", "1")))


def _generation_batch_size(
    *,
    per_device_batch: int,
    grad_accum: int,
    num_generations: int,
    world_size: int,
    override: int | None = None,
) -> int:
    """Scheme A: scale rollout batch with GPU parallelism to keep prompt/step high."""
    batch_size = override or (per_device_batch * world_size * grad_accum)
    if batch_size % num_generations != 0:
        raise ValueError(
            f"generation_batch_size ({batch_size}) must be divisible by num_generations ({num_generations})"
        )
    global_batch = per_device_batch * world_size
    if batch_size % global_batch != 0:
        raise ValueError(
            f"generation_batch_size ({batch_size}) must be divisible by global batch ({global_batch})"
        )
    return batch_size


def _configure_fla() -> None:
    """Use PyTorch linear-attn fallback unless USE_FLA=1.

    On H100, FLA's TileLang backend may fail to JIT-compile (ptxas error), and
    Triton >= 3.4 is blocked on Hopper without TileLang (fla #640).
    """
    if os.environ.get("USE_FLA", "0") == "1":
        os.environ.setdefault("FLA_TILELANG", "0")
        return

    import transformers.utils.import_utils as import_utils

    if hasattr(import_utils.is_flash_linear_attention_available, "cache_clear"):
        import_utils.is_flash_linear_attention_available.cache_clear()
    import_utils.is_flash_linear_attention_available = lambda: False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/Qwen3.5-2B")
    parser.add_argument("--train_path", type=str, default="data/processed/mbpp_train")
    parser.add_argument("--output_dir", type=str, default="outputs/grpo-mbpp")
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        default=None,
        help="Rollout batch size; default = per_device_batch × num_gpus × grad_accum",
    )
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--swanlab_project", type=str, default="coding-rl")
    parser.add_argument("--swanlab_experiment_name", type=str, default="grpo-mbpp")
    parser.add_argument("--swanlab_description", type=str, default="GRPO on MBPP with code execution reward")
    parser.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging")
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="KL penalty vs reference model; 0 disables ref model and KL term",
    )
    parser.add_argument(
        "--full_finetune",
        action="store_true",
        help="Train all model weights (no LoRA). Requires more GPU memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_fla()

    train_dataset = load_from_disk(args.train_path)

    peft_config = None if args.full_finetune else _build_lora_config(args.model_path, r=32, alpha=64)
    if args.full_finetune:
        print("Training mode: full fine-tuning (peft_config=None)")
    else:
        print("Training mode: LoRA (r=32, alpha=64)")

    world_size = _world_size()
    generation_batch_size = _generation_batch_size(
        per_device_batch=args.per_device_train_batch_size,
        grad_accum=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        world_size=world_size,
        override=args.generation_batch_size,
    )
    prompts_per_generate = generation_batch_size // args.num_generations
    print(
        f"Scheme A: {world_size} GPU(s), generation_batch_size={generation_batch_size}, "
        f"prompts/generate={prompts_per_generate}, num_generations={args.num_generations}, "
        f"beta={args.beta}, lr={args.learning_rate}"
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        generation_batch_size=generation_batch_size,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        chat_template_kwargs={"enable_thinking": False},
        reward_weights=[0.1, 0.9],
        use_vllm=False,
        beta=args.beta,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        ddp_find_unused_parameters=True if args.full_finetune else None,
    )

    callbacks = []
    if not args.no_swanlab:
        callbacks.append(
            SwanLabCallback(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment_name,
                description=args.swanlab_description,
            )
        )

    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=[format_reward, execution_reward],
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
