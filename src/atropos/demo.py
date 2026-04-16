"""One-command demo workflow for Atropos."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from atropos.rl_env.line_world import LineWorldEnv


@dataclass(frozen=True)
class DemoConfig:
    """Minimal config for `atropos demo`."""

    api_host: str = "127.0.0.1"
    api_port: int = 8011
    trainer_steps: int = 12
    step_sleep_seconds: float = 0.2
    environment_goal: int = 6
    environment_max_steps: int = 20
    environment_seed: int = 7

    @classmethod
    def from_yaml(cls, path: Path) -> DemoConfig:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Demo config must be a YAML mapping.")

        config = cls(
            api_host=str(data.get("api_host", cls.api_host)),
            api_port=int(data.get("api_port", cls.api_port)),
            trainer_steps=int(data.get("trainer_steps", cls.trainer_steps)),
            step_sleep_seconds=float(data.get("step_sleep_seconds", cls.step_sleep_seconds)),
            environment_goal=int(data.get("environment_goal", cls.environment_goal)),
            environment_max_steps=int(data.get("environment_max_steps", cls.environment_max_steps)),
            environment_seed=int(data.get("environment_seed", cls.environment_seed)),
        )
        if config.trainer_steps <= 0:
            raise ValueError("trainer_steps must be > 0")
        if config.api_port <= 0:
            raise ValueError("api_port must be > 0")
        if config.step_sleep_seconds < 0:
            raise ValueError("step_sleep_seconds must be >= 0")
        return config


class _ServerHandle:
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._server: Any | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "uvicorn is required for `atropos demo`. Install with `pip install uvicorn`."
            ) from exc
        try:
            from atroposlib.api.server import HardeningTier, build_runtime_app
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "fastapi is required for `atropos demo`. Install with `pip install fastapi`."
            ) from exc

        app = build_runtime_app(tier=HardeningTier.RESEARCH_SAFE)
        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

        wait_deadline = time.time() + 5.0
        while time.time() < wait_deadline:
            if getattr(self._server, "started", False):
                return
            time.sleep(0.05)
        raise RuntimeError("Timed out starting demo API server.")

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5.0)


def _enqueue_job(base_url: str, step: int) -> dict[str, Any]:
    payload = json.dumps({"step": step, "kind": "demo"}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/jobs",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3.0) as response:
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("Invalid response from demo API server.")
    return parsed


def run_demo(config_path: Path) -> int:
    """Execute a local, observable end-to-end demo."""

    config = DemoConfig.from_yaml(config_path)
    base_url = f"http://{config.api_host}:{config.api_port}"
    server = _ServerHandle(host=config.api_host, port=config.api_port)

    print("Starting Atropos demo...\n")
    print(f"[1/4] API server: {base_url}")
    server.start()

    print("[2/4] Environment: LineWorld")
    env = LineWorldEnv(
        goal=config.environment_goal,
        max_steps=config.environment_max_steps,
        seed=config.environment_seed,
    )
    env.reset()

    print(f"[3/4] Trainer loop: {config.trainer_steps} simulated iterations")
    print("[4/4] Live metrics:")
    print("step | reward | position | queue_depth")

    total_reward = 0.0
    completed = 0
    try:
        for step in range(1, config.trainer_steps + 1):
            record = env.step(action=1)
            api_result = _enqueue_job(base_url, step)
            queue_depth = int(api_result.get("queue_depth", 0))
            total_reward += record.reward
            completed = step
            metrics_line = (
                f"{step:>4} | {record.reward:>6.2f} | "
                f"{record.position_after:>8} | {queue_depth:>11}"
            )
            print(metrics_line)
            if record.done:
                env.reset()
            time.sleep(config.step_sleep_seconds)
    finally:
        server.stop()

    avg_reward = total_reward / max(1, completed)
    print("\nDemo complete ✅")
    print(
        f"Summary: iterations={completed}, avg_reward={avg_reward:.2f}, "
        f"final_queue_depth={queue_depth}"
    )
    return 0
