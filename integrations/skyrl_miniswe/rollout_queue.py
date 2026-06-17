"""HTTP rollout job queue (GPU side) + async pull client for the generator."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_QUEUE_HOST = os.environ.get("SKYRL_ROLLOUT_QUEUE_HOST", "127.0.0.1")
DEFAULT_QUEUE_PORT = int(os.environ.get("SKYRL_ROLLOUT_QUEUE_PORT", "9000"))


def rollout_queue_base_url() -> str:
    return os.environ.get(
        "SKYRL_ROLLOUT_QUEUE_URL",
        f"http://{DEFAULT_QUEUE_HOST}:{DEFAULT_QUEUE_PORT}",
    ).rstrip("/")


def use_pull_rollout() -> bool:
    return os.environ.get("SKYRL_ROLLOUT_MODE", "pull").lower() != "ray"


@dataclass
class RolloutJob:
    job_id: str
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass
class RolloutResult:
    messages: list
    reward: int
    error: str | None


class RolloutQueueState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: deque[RolloutJob] = deque()
        self._results: dict[str, RolloutResult] = {}
        self._waiters: dict[str, list[threading.Event]] = {}
        self._dequeue_waiters: list[threading.Condition] = []

    def enqueue(self, payload: dict[str, Any]) -> str:
        job_id = uuid.uuid4().hex
        with self._lock:
            self._pending.append(RolloutJob(job_id=job_id, payload=payload))
            for cond in self._dequeue_waiters:
                cond.notify()
        return job_id

    def dequeue(self, timeout: float) -> RolloutJob | None:
        deadline = time.time() + timeout
        cond = threading.Condition(self._lock)
        with cond:
            self._dequeue_waiters.append(cond)
            try:
                while not self._pending:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    cond.wait(timeout=remaining)
                return self._pending.popleft()
            finally:
                if cond in self._dequeue_waiters:
                    self._dequeue_waiters.remove(cond)

    def submit_result(self, job_id: str, messages: list, reward: int, error: str | None) -> bool:
        with self._lock:
            result = RolloutResult(messages=messages, reward=reward, error=error)
            self._results[job_id] = result
            for event in self._waiters.pop(job_id, []):
                event.set()
        return True

    def wait_result(self, job_id: str, timeout: float) -> RolloutResult | None:
        deadline = time.time() + timeout
        with self._lock:
            if job_id in self._results:
                return self._results.pop(job_id)
            event = threading.Event()
            self._waiters.setdefault(job_id, []).append(event)

        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                with self._lock:
                    waiters = self._waiters.get(job_id, [])
                    if event in waiters:
                        waiters.remove(event)
                    if job_id in self._results:
                        return self._results.pop(job_id)
                return None
            if event.wait(timeout=min(remaining, 1.0)):
                with self._lock:
                    if job_id in self._results:
                        return self._results.pop(job_id)
                return None


_QUEUE_STATE = RolloutQueueState()


def get_queue_state() -> RolloutQueueState:
    return _QUEUE_STATE


class _RolloutQueueHandler(BaseHTTPRequestHandler):
    server_version = "SkyRLRolloutQueue/1.0"

    @property
    def state(self) -> RolloutQueueState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        if os.environ.get("SKYRL_ROLLOUT_QUEUE_QUIET", "1") != "0":
            return
        super().log_message(format, *args)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _send_json(self, status: int, payload: dict[str, Any] | None = None) -> None:
        body = b""
        if payload is not None:
            body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        query = parse_qs(parsed.query)

        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        if path == "/v1/dequeue":
            timeout = float(query.get("timeout", ["30"])[0])
            worker_id = query.get("worker_id", ["unknown"])[0]
            job = self.state.dequeue(timeout=timeout)
            if job is None:
                self._send_json(204)
                return
            self._send_json(
                200,
                {
                    "job_id": job.job_id,
                    "worker_id": worker_id,
                    "payload": job.payload,
                },
            )
            return

        if path.startswith("/v1/jobs/") and path.endswith("/wait"):
            job_id = path.split("/")[3]
            timeout = float(query.get("timeout", ["3600"])[0])
            result = self.state.wait_result(job_id, timeout=timeout)
            if result is None:
                self._send_json(504, {"error": "timeout waiting for rollout result"})
                return
            self._send_json(
                200,
                {
                    "messages": result.messages,
                    "reward": result.reward,
                    "error": result.error,
                },
            )
            return

        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/v1/jobs":
            body = self._read_json()
            job_id = self.state.enqueue(body)
            self._send_json(200, {"job_id": job_id})
            return

        if path.startswith("/v1/jobs/") and path.endswith("/result"):
            job_id = path.split("/")[3]
            body = self._read_json()
            self.state.submit_result(
                job_id,
                messages=body.get("messages", []),
                reward=int(body.get("reward", 0)),
                error=body.get("error"),
            )
            self._send_json(200, {"ok": True})
            return

        self._send_json(404, {"error": "not found"})


@dataclass
class RolloutQueueServer:
    host: str
    port: int
    httpd: ThreadingHTTPServer
    thread: threading.Thread

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.thread.join(timeout=5)


def start_rollout_queue_server(
    host: str | None = None,
    port: int | None = None,
) -> RolloutQueueServer:
    host = host or DEFAULT_QUEUE_HOST
    port = port or DEFAULT_QUEUE_PORT
    httpd = ThreadingHTTPServer((host, port), _RolloutQueueHandler)
    httpd.state = _QUEUE_STATE  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, name="rollout-queue", daemon=True)
    thread.start()
    return RolloutQueueServer(host=host, port=port, httpd=httpd, thread=thread)


async def _http_request(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 3600.0,
) -> tuple[int, dict[str, Any] | None]:
    def _run() -> tuple[int, dict[str, Any] | None]:
        data = None
        headers = {"Content-Type": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
        import urllib.error
        import urllib.request

        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
                if not raw:
                    return resp.status, None
                return resp.status, json.loads(raw)
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            payload = json.loads(raw) if raw else None
            return exc.code, payload

    return await asyncio.to_thread(_run)


def _trajectory_id_to_dict(trajectory_id) -> dict[str, Any]:
    return {
        "instance_id": trajectory_id.instance_id,
        "repetition_id": trajectory_id.repetition_id,
    }


def _trajectory_id_from_dict(data: dict[str, Any]):
    from integrations.skyrl_miniswe.types import TrajectoryID

    return TrajectoryID(
        instance_id=data["instance_id"],
        repetition_id=int(data["repetition_id"]),
    )


async def run_rollout_job(
    instance: dict,
    litellm_model_name: str,
    sweagent_config: dict,
    miniswe_traj_dir: str,
    data_source: str,
    sampling_params: dict,
    trajectory_id,
    global_step: int,
    training_phase: str,
) -> tuple[list, int, str | None]:
    base = rollout_queue_base_url()
    payload = {
        "instance": instance,
        "litellm_model_name": litellm_model_name,
        "sweagent_config": sweagent_config,
        "miniswe_traj_dir": miniswe_traj_dir,
        "data_source": data_source,
        "sampling_params": sampling_params,
        "trajectory_id": _trajectory_id_to_dict(trajectory_id),
        "global_step": global_step,
        "training_phase": training_phase,
    }
    status, created = await _http_request("POST", f"{base}/v1/jobs", body=payload, timeout=60.0)
    if status != 200 or not created:
        raise RuntimeError(f"failed to enqueue rollout job: status={status} body={created}")
    job_id = created["job_id"]
    wait_timeout = float(os.environ.get("SKYRL_ROLLOUT_JOB_TIMEOUT_S", "7200"))
    status, result = await _http_request(
        "GET",
        f"{base}/v1/jobs/{job_id}/wait?timeout={wait_timeout}",
        timeout=wait_timeout + 30.0,
    )
    if status != 200 or not result:
        raise RuntimeError(f"rollout job {job_id} failed: status={status} body={result}")
    return result.get("messages", []), int(result.get("reward", 0)), result.get("error")


def run_pull_worker_loop(
    queue_url: str | None = None,
    worker_id: str | None = None,
    dequeue_timeout: float = 30.0,
) -> None:
    """Blocking loop for CPU-side rollout workers."""
    from integrations.skyrl_miniswe.rollout_core import init_and_run

    base = (queue_url or rollout_queue_base_url()).rstrip("/")
    worker_id = worker_id or f"{os.uname().nodename}-{os.getpid()}"
    print(f"==> Pull rollout worker {worker_id} -> {base}", flush=True)

    while True:
        import urllib.error
        import urllib.request

        url = f"{base}/v1/dequeue?timeout={dequeue_timeout}&worker_id={worker_id}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=dequeue_timeout + 10.0) as resp:
                if resp.status == 204:
                    continue
                job = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 204:
                continue
            raise

        job_id = job["job_id"]
        payload = job["payload"]
        trajectory_id = _trajectory_id_from_dict(payload["trajectory_id"])
        messages, reward, error = init_and_run(
            instance=payload["instance"],
            litellm_model_name=payload["litellm_model_name"],
            sweagent_config=payload["sweagent_config"],
            miniswe_traj_dir=payload["miniswe_traj_dir"],
            data_source=payload["data_source"],
            sampling_params=payload["sampling_params"],
            trajectory_id=trajectory_id,
            global_step=int(payload["global_step"]),
            training_phase=payload["training_phase"],
        )
        result_body = json.dumps(
            {"messages": messages, "reward": reward, "error": error},
        ).encode()
        post_req = urllib.request.Request(
            f"{base}/v1/jobs/{job_id}/result",
            data=result_body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(post_req, timeout=60.0) as resp:
            resp.read()
        print(f"==> Completed job {job_id} reward={reward} error={error!r}", flush=True)
