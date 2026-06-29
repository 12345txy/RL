#!/usr/bin/env python3
"""Multi-turn SFT for Gemma4 on mini-swe-agent/SWE trajectories."""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from swanlab.integration.transformers import SwanLabCallback
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.chunked_nll_deepspeed import patch_chunked_nll_for_deepspeed_zero3
from training.gemma_lora import build_lora_config
from training.sft_preprocess_cache import is_main_process, prepare_sft_dataset


def _configure_torch_backends() -> None:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.set_float32_matmul_precision("high")


def _configure_fla() -> None:
    if os.environ.get("USE_FLA", "0") == "1":
        return
    import transformers.utils.import_utils as import_utils

    if hasattr(import_utils.is_flash_linear_attention_available, "cache_clear"):
        import_utils.is_flash_linear_attention_available.cache_clear()
    import_utils.is_flash_linear_attention_available = lambda: False


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", os.environ.get("ACCELERATE_NUM_PROCESSES", "1")))


def _is_rank0() -> bool:
    return is_main_process()


def _print_training_plan(args: argparse.Namespace, num_samples: int) -> None:
    if not _is_rank0():
        return
    global_batch = args.per_device_train_batch_size * _world_size() * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(num_samples / global_batch)
    total_steps = steps_per_epoch * args.num_train_epochs
    print("==> Training plan")
    print(f"    samples={num_samples} epochs={args.num_train_epochs}")
    print(
        f"    per_device_batch={args.per_device_train_batch_size} "
        f"grad_accum={args.gradient_accumulation_steps} gpus={_world_size()}"
    )
    print(f"    global_batch={global_batch} steps_per_epoch={steps_per_epoch} total_steps={total_steps}")
    print(f"    max_seq_length={args.max_seq_length} log_every={args.logging_steps} step(s)")
    if args.max_seq_length >= 16384:
        print("    note: first step with long context can take several minutes before metrics appear")


class StepProgressCallback(TrainerCallback):
    """Print step timing on rank 0 so long steps are visibly alive."""

    def __init__(self) -> None:
        self._step_t0 = time.time()
        self._train_t0 = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not _is_rank0() or not logs:
            return
        elapsed = time.time() - self._step_t0
        total = time.time() - self._train_t0
        metrics = ", ".join(f"{k}={v}" for k, v in logs.items() if k != "epoch")
        print(
            f"[step {state.global_step}/{state.max_steps}] "
            f"step_time={elapsed:.1f}s elapsed={total/60:.1f}min | {metrics}",
            flush=True,
        )




def _model_init_kwargs() -> dict[str, object]:
    return {
        "trust_remote_code": True,
        "torch_dtype": "bfloat16",
        "attn_implementation": "sdpa",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="models/gemma-4-12B-it")
    parser.add_argument("--train_path", default="data/sft/sft_merged.jsonl")
    parser.add_argument("--output_dir", default="outputs/sft-gemma4-12b-miniswe")
    parser.add_argument("--max_seq_length", type=int, default=40960)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument(
        "--deepspeed_zero_stage",
        type=int,
        default=3,
        choices=(1, 2, 3),
        help="DeepSpeed ZeRO stage (must match ACCEL_CONFIG); stage-3 chunked_nll needs lm_head gather patch",
    )
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--logging_steps", type=int, default=1, help="Log metrics every N optimizer steps")
    parser.add_argument(
        "--ddp_timeout",
        type=int,
        default=1200,
        help="NCCL collective timeout in seconds (default: 1200 = 20 minutes)",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--swanlab_project", type=str, default="swe-rl")
    parser.add_argument("--swanlab_experiment_name", type=str, default="sft-gemma4-12b")
    parser.add_argument("--swanlab_description", type=str, default="Gemma4-12B SFT on mini-swe-agent/SWE trajectories")
    parser.add_argument("--no_swanlab", action="store_true", help="Disable SwanLab logging")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="chunked_nll",
        choices=("nll", "chunked_nll", "dft"),
        help="chunked_nll avoids materializing full vocab logits (required for 32k ctx)",
    )
    parser.add_argument(
        "--truncation_policy",
        type=str,
        default="chunk",
        choices=("none", "smart_tail", "chunk"),
        help="Agent-aware context limiting before tokenization (default: chunk by turns)",
    )
    parser.add_argument(
        "--preprocess_cache_dir",
        type=str,
        default="data/sft/preprocessed",
        help="Base directory for on-disk preprocessed (tokenized) datasets",
    )
    parser.add_argument(
        "--use_preprocessed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load from disk cache when .ready marker and manifest match (default: true)",
    )
    parser.add_argument(
        "--force_preprocess",
        action="store_true",
        help="Rebuild preprocessed cache even if a valid .ready marker exists",
    )
    parser.add_argument(
        "--preprocess_num_proc",
        type=int,
        default=8,
        help="CPU workers for cache build on rank 0 (tokenization)",
    )
    parser.add_argument(
        "--preprocess_only",
        action="store_true",
        help="Only build/load preprocessed cache, then exit without training",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade compute for memory; disable only if you have headroom (default: true)",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        nargs="?",
        const="latest",
        default=None,
        metavar="CHECKPOINT",
        help="Resume training: omit value for latest checkpoint in output_dir, or pass a checkpoint path",
    )
    return parser.parse_args()


def _resolve_resume_checkpoint(value: str | None, output_dir: str) -> bool | str | None:
    if value is None:
        return None
    if value.lower() in ("true", "1", "yes", "latest"):
        return True
    path = Path(value)
    if not path.is_absolute():
        path = Path(output_dir) / path
    if not path.is_dir():
        raise SystemExit(f"Resume checkpoint not found: {path}")
    return str(path)


def main() -> None:
    args = parse_args()
    _configure_fla()
    _configure_torch_backends()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset, cache_dir, from_cache = prepare_sft_dataset(
        train_path=args.train_path,
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
        max_samples=args.max_samples,
        tokenizer=tokenizer,
        cache_base_dir=args.preprocess_cache_dir,
        use_preprocessed=args.use_preprocessed,
        force_preprocess=args.force_preprocess,
        preprocess_num_proc=args.preprocess_num_proc,
        truncation_policy=args.truncation_policy,
        is_main_process=_is_rank0(),
    )
    num_samples = len(dataset)

    if _is_rank0():
        print(f"==> SFT dataset: {num_samples} rows from {args.train_path}")
        print(
            f"    cache={cache_dir} from_disk={from_cache} "
            f"world_size={_world_size()} full_finetune={args.full_finetune} "
            f"loss_type={args.loss_type} truncation_policy={args.truncation_policy}"
        )

    if args.preprocess_only:
        if _is_rank0():
            print(f"==> Preprocess-only complete: {cache_dir}")
        return

    peft_config = None if args.full_finetune else build_lora_config(
        args.model_path, r=args.lora_r, alpha=args.lora_alpha
    )

    if args.loss_type == "chunked_nll" and args.deepspeed_zero_stage == 3 and _world_size() > 1:
        patch_chunked_nll_for_deepspeed_zero3()

    model_kwargs = _model_init_kwargs()
    # ZeRO-3 / multi-GPU: pass path so weights are sharded at init, not replicated per rank.
    if args.full_finetune and _world_size() == 1:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    else:
        model = args.model_path

    global_batch = args.per_device_train_batch_size * _world_size() * args.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(num_samples / global_batch))
    total_steps = steps_per_epoch * args.num_train_epochs
    warmup_steps = max(1, int(total_steps * 0.03))

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        bf16=True,
        optim="adamw_torch_fused",
        model_init_kwargs=model_kwargs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_first_step=True,
        log_level="info",
        log_level_replica="warning",
        report_to="none",
        packing=False,
        loss_type=args.loss_type,
        ddp_timeout=args.ddp_timeout,
    )

    _print_training_plan(args, num_samples)

    callbacks = [StepProgressCallback()]
    if not args.no_swanlab:
        callbacks.append(
            SwanLabCallback(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment_name,
                description=args.swanlab_description,
            )
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    resume = _resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    if _is_rank0() and resume is not None:
        label = "latest in output_dir" if resume is True else resume
        print(f"==> Resuming from checkpoint: {label}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"==> Saved checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
