"""Runtime worker process with readiness/liveness endpoints.

Stability tier: Tier 1 (platform core runtime worker).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request, urlopen

import uvicorn
from fastapi import FastAPI, Response, status

LOGGER = logging.getLogger("atroposlib.workers.runtime")


@dataclass(slots=True)
class WorkerHealthState:
    """Mutable health state reflected by worker probes."""

    started: bool = False
    ready: bool = False
    shutdown: bool = False
    last_check_at: str | None = None
    last_error: str | None = None


def _get_json(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, method="GET")
    with urlopen(request, timeout=timeout) as response:  # nosec: B310
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object response from runtime API")
    return cast(dict[str, Any], payload)


class RuntimeWorker:
    """Polls runtime API readiness as a dependency gate for worker startup."""

    def __init__(
        self,
        *,
        api_base_url: str,
        poll_interval_seconds: float,
        request_timeout_seconds: float,
    ) -> None:
        self.api_base_url = api_base_url.rstrip("/")
        self.poll_interval_seconds = poll_interval_seconds
        self.request_timeout_seconds = request_timeout_seconds
        self.health_state = WorkerHealthState()

    def _check_api_ready(self) -> None:
        readiness = _get_json(
            f"{self.api_base_url}/health/ready",
            timeout=self.request_timeout_seconds,
        )
        if readiness.get("status") != "ready":
            raise RuntimeError(f"Runtime API not ready: {readiness}")

    async def run(self) -> None:
        self.health_state.started = True
        while not self.health_state.shutdown:
            try:
                self._check_api_ready()
            except (RuntimeError, ValueError, URLError, TimeoutError) as exc:
                self.health_state.ready = False
                self.health_state.last_error = str(exc)
                LOGGER.warning("runtime_worker_dependency_unready", extra={"error": str(exc)})
            else:
                self.health_state.ready = True
                self.health_state.last_error = None
                LOGGER.info("runtime_worker_dependency_ready")
            finally:
                self.health_state.last_check_at = datetime.now(timezone.utc).isoformat()
            await asyncio.sleep(self.poll_interval_seconds)


def build_worker_app(worker: RuntimeWorker) -> FastAPI:
    """Create FastAPI health app bound to a runtime worker loop."""

    @asynccontextmanager
    async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
        task = asyncio.create_task(worker.run())
        try:
            yield
        finally:
            worker.health_state.shutdown = True
            await task

    app = FastAPI(title="Atropos Runtime Worker", lifespan=_lifespan)

    def livez(response: Response) -> dict[str, Any]:
        state = worker.health_state
        if state.shutdown:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "terminating", "last_check_at": state.last_check_at}
        if not state.started:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return {"status": "starting"}
        response.status_code = status.HTTP_200_OK
        return {"status": "alive", "last_check_at": state.last_check_at}

    def readyz(response: Response) -> dict[str, Any]:
        state = worker.health_state
        if state.ready and not state.shutdown:
            response.status_code = status.HTTP_200_OK
            return {"status": "ready", "last_check_at": state.last_check_at}
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "last_check_at": state.last_check_at,
            "reason": state.last_error,
        }

    app.add_api_route("/livez", livez, methods=["GET"])
    app.add_api_route("/readyz", readyz, methods=["GET"])
    return app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atropos runtime worker process")
    parser.add_argument(
        "--host",
        default=os.getenv("WORKER_HOST", "0.0.0.0"),
        help="Bind host for worker health endpoints.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("WORKER_PORT", "9000")),
        help="Bind port for worker health endpoints.",
    )
    parser.add_argument(
        "--api-base-url",
        default=os.getenv("ATROPOS_API_BASE_URL", "http://atropos-api:8000"),
        help="Base URL for runtime API dependency checks.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=float(os.getenv("WORKER_POLL_INTERVAL_SECONDS", "5")),
        help="Seconds between dependency readiness checks.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=float(os.getenv("WORKER_REQUEST_TIMEOUT_SECONDS", "2")),
        help="HTTP timeout in seconds for dependency checks.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=os.getenv("ATROPOS_LOG_LEVEL", "INFO"))
    args = _build_parser().parse_args()
    worker = RuntimeWorker(
        api_base_url=args.api_base_url,
        poll_interval_seconds=args.poll_interval_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    app = build_worker_app(worker)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
