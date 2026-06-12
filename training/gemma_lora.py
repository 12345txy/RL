"""Shared Gemma4 LoRA configuration helpers."""

from __future__ import annotations

from pathlib import Path

from peft import LoraConfig

DEFAULT_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def gemma4_language_lora_targets(num_layers: int) -> list[str]:
    targets: list[str] = []
    for layer_idx in range(num_layers):
        prefix = f"model.language_model.layers.{layer_idx}"
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            targets.append(f"{prefix}.self_attn.{name}")
        for name in ("gate_proj", "up_proj", "down_proj"):
            targets.append(f"{prefix}.mlp.{name}")
    return targets


def build_lora_config(model_path: str, *, r: int = 64, alpha: int = 128) -> LoraConfig:
    from transformers import AutoConfig

    target_modules = DEFAULT_LORA_TARGETS
    model_name = Path(model_path).name.lower()
    if "gemma" in model_name:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        num_layers = getattr(text_config, "num_hidden_layers", None)
        if num_layers is None:
            raise ValueError(f"Cannot infer num_hidden_layers for Gemma model: {model_path}")
        target_modules = gemma4_language_lora_targets(num_layers)

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
