"""Resilience primitives for retries and process isolation."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable
from dataclasses import dataclass
from queue import Empty
from typing import Any, TypeVar, cast

from .error_categories import is_recoverable

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Retry policy for transient failures."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 2.0


@dataclass(frozen=True)
class RetryResult:
    """Result value and number of retries consumed."""

    value: Any
    retry_count: int


def retry_call(fn: Callable[[], T], retry_policy: RetryPolicy) -> RetryResult:
    """Execute function with exponential backoff on recoverable errors."""
    attempts = 0
    while True:
        try:
            value = fn()
            return RetryResult(value=value, retry_count=attempts)
        except Exception as exc:  # noqa: BLE001 - intentionally catches task failures
            attempts += 1
            if attempts >= retry_policy.max_attempts or not is_recoverable(exc):
                raise
            delay = min(
                retry_policy.max_delay_seconds,
                retry_policy.base_delay_seconds * (2 ** (attempts - 1)),
            )
            time.sleep(delay)


def _process_entrypoint(
    queue: mp.Queue[Any], fn: Callable[..., Any], kwargs: dict[str, Any]
) -> None:
    try:
        queue.put(("ok", fn(**kwargs)))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", exc))


def run_with_timeout(fn: Callable[..., T], timeout_seconds: int, **kwargs: Any) -> T:
    """Run work in an isolated subprocess and kill it on timeout."""
    queue: mp.Queue[Any] = mp.Queue(maxsize=1)
    process = mp.Process(target=_process_entrypoint, args=(queue, fn, kwargs))
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.kill()
        process.join(timeout=1)
        raise TimeoutError(f"Scenario timed out after {timeout_seconds} seconds")

    try:
        status, payload = queue.get_nowait()
    except Empty as exc:
        raise RuntimeError("Scenario worker exited without returning a result") from exc

    if status == "error":
        if isinstance(payload, Exception):
            raise payload
        raise RuntimeError(str(payload))
    return cast(T, payload)
