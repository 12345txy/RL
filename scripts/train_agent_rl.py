#!/usr/bin/env python3
"""DEPRECATED: Legacy pseudo-RL (mock env + SFT refresh). Use SkyRL: bash scripts/run_rl_skyrl.sh"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from integrations.skyrl_miniswe.generator import MiniSweRolloutGenerator, load_rl_pool
from data.swe_utils import load_json, save_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rl_pool", default="data/rl/swegym_rl_train.jsonl")
    parser.add_argument("--output_dir", default="outputs/rl-gemma4-12b-miniswe")
    parser.add_argument("--api_base", default=os.environ.get("VLLM_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default="gemma-4-12B-it")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_rollouts", type=int, default=4)
    parser.add_argument("--max_turns", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--step_penalty", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev_split", default="data/splits/verified_dev_100.json")
    parser.add_argument("--save_every", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pool = load_rl_pool(args.rl_pool)
    if not pool:
        raise SystemExit(f"Empty RL pool: {args.rl_pool}")

    out = Path(args.output_dir)
    rollouts_dir = out / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    generator = MiniSweRolloutGenerator(
        api_base=args.api_base,
        model=args.model,
        max_turns=args.max_turns,
        temperature=args.temperature,
        step_penalty=args.step_penalty,
    )

    metrics = {"steps": [], "mean_reward": [], "resolved": []}
    sft_buffer: list[dict] = []

    for step in range(1, args.steps + 1):
        batch = [pool[rng.randrange(len(pool))] for _ in range(args.batch_size)]
        step_rewards: list[float] = []
        step_resolved = 0

        for inst in batch:
            group_rewards: list[float] = []
            group_rollouts = []
            for _ in range(args.num_rollouts):
                result = generator.rollout(inst)
                group_rollouts.append(result)
                group_rewards.append(result.reward)
                if result.reward >= 1.0:
                    step_resolved += 1

            mean_r = sum(group_rewards) / len(group_rewards)
            for rollout, reward in zip(group_rollouts, group_rewards):
                advantage = reward - mean_r
                if advantage > 0 or reward >= 1.0:
                    sft_buffer.append(
                        {
                            "instance_id": rollout.instance_id,
                            "messages": rollout.messages,
                            "source": "agent_rl",
                            "metadata": {
                                "reward": reward,
                                "advantage": advantage,
                                "step": step,
                            },
                        }
                    )
                step_rewards.append(reward)

        mean_reward = sum(step_rewards) / max(len(step_rewards), 1)
        metrics["steps"].append(step)
        metrics["mean_reward"].append(mean_reward)
        metrics["resolved"].append(step_resolved)
        print(f"step={step} mean_reward={mean_reward:.4f} resolved_count={step_resolved}")

        if step % args.save_every == 0 or step == args.steps:
            save_jsonl(rollouts_dir / f"sft_buffer_step{step}.jsonl", sft_buffer)
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Write Sky-RL / SFT continuation dataset
    final_sft = out / "rl_positive_trajectories.jsonl"
    save_jsonl(final_sft, sft_buffer)
    print(f"==> Saved {len(sft_buffer)} positive/advantaged trajectories -> {final_sft}")
    print("==> Continue policy update with:")
    print(f"    TRAIN_PATH={final_sft} STAGE=lora bash scripts/run_sft.sh")


if __name__ == "__main__":
    main()
