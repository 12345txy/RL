#!/usr/bin/env python3
"""CPU-side pull worker for Mini-SWE Docker rollouts."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull Mini-SWE rollout jobs from GPU queue")
    parser.add_argument(
        "--queue-url",
        default=os.environ.get("SKYRL_ROLLOUT_QUEUE_URL", "http://127.0.0.1:9000"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SKYRL_ROLLOUT_PULL_WORKERS", "4")),
    )
    parser.add_argument(
        "--dequeue-timeout",
        type=float,
        default=float(os.environ.get("SKYRL_ROLLOUT_DEQUEUE_TIMEOUT_S", "30")),
    )
    args = parser.parse_args()

    if args.workers <= 1:
        from integrations.skyrl_miniswe.rollout_queue import run_pull_worker_loop

        run_pull_worker_loop(queue_url=args.queue_url, dequeue_timeout=args.dequeue_timeout)
        return

    processes: list[mp.Process] = []
    for idx in range(args.workers):
        worker_id = f"{os.uname().nodename}-{os.getpid()}-{idx}"
        proc = mp.Process(
            target=_worker_entry,
            args=(args.queue_url, worker_id, args.dequeue_timeout),
            daemon=False,
        )
        proc.start()
        processes.append(proc)

    for proc in processes:
        proc.join()


def _worker_entry(queue_url: str, worker_id: str, dequeue_timeout: float) -> None:
    from integrations.skyrl_miniswe.rollout_queue import run_pull_worker_loop

    run_pull_worker_loop(queue_url=queue_url, worker_id=worker_id, dequeue_timeout=dequeue_timeout)


if __name__ == "__main__":
    main()
