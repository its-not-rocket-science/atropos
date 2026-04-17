from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pytest

from atroposlib.envs.transport_client import (
    TransportClient,
    TransportClientError,
    TransportMalformedResponseError,
    TransportRetriesExhaustedError,
    TransportTimeoutError,
)


class FakeHttpError(RuntimeError):
    def __init__(self, status_code: int, message: str = "http error") -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class SequencedTransport(TransportClient):
    responses: deque[Any] = field(default_factory=deque)
    seen_payloads: list[dict[str, Any]] = field(default_factory=list)

    def _send_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.seen_payloads.append(dict(payload))
        value = self.responses.popleft()
        if isinstance(value, Exception):
            raise value
        return value


def test_transport_retries_timeout_with_bounded_backoff() -> None:
    sleeps: list[float] = []
    transport = SequencedTransport(
        responses=deque(
            [
                TimeoutError("first timeout"),
                TimeoutError("second timeout"),
                {"ok": True, "value": 7},
            ]
        ),
        max_retries=3,
        base_backoff_seconds=0.1,
        backoff_multiplier=2.0,
        max_backoff_seconds=0.15,
        sleep_fn=sleeps.append,
        request_id_factory=lambda: "req-timeout",
    )

    response = transport.send({"env": "toy"})

    assert response["ok"] is True
    assert response["request_id"] == "req-timeout"
    assert response["retry_count"] == 2
    assert sleeps == [0.1, 0.15]
    assert transport.metrics["transport_retries_total"] == 2
    assert transport.metrics["transport_retry_count_total"] == 2


def test_transport_retries_5xx_then_raises_terminal_failure() -> None:
    sleeps: list[float] = []
    transport = SequencedTransport(
        responses=deque(
            [
                FakeHttpError(503, "unavailable"),
                FakeHttpError(502, "gateway"),
                FakeHttpError(500, "still down"),
            ]
        ),
        max_retries=2,
        sleep_fn=sleeps.append,
        request_id_factory=lambda: "req-5xx",
    )

    with pytest.raises(TransportRetriesExhaustedError) as exc_info:
        transport.send({"env": "toy"})

    assert exc_info.value.request_id == "req-5xx"
    assert exc_info.value.retry_count == 2
    assert "Transport retries exhausted" in str(exc_info.value)
    assert sleeps == [0.05, 0.1]
    assert transport.metrics["transport_terminal_failures_total"] == 1


def test_transport_4xx_is_non_retryable_and_fails_fast() -> None:
    sleeps: list[float] = []
    transport = SequencedTransport(
        responses=deque([FakeHttpError(400, "bad request")]),
        max_retries=5,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(TransportClientError):
        transport.send({"env": "toy"})

    assert sleeps == []
    assert transport.metrics["transport_retries_total"] == 0
    assert transport.metrics["transport_terminal_failures_total"] == 1


def test_transport_malformed_response_is_non_retryable() -> None:
    sleeps: list[float] = []
    transport = SequencedTransport(
        responses=deque([{"unexpected": True}]),
        max_retries=4,
        sleep_fn=sleeps.append,
    )

    with pytest.raises(TransportMalformedResponseError):
        transport.send({"env": "toy"})

    assert sleeps == []
    assert transport.metrics["transport_terminal_failures_total"] == 1


def test_transport_preserves_request_id_across_retries() -> None:
    transport = SequencedTransport(
        responses=deque([TransportTimeoutError("timeout"), {"ok": True}]),
        max_retries=1,
        sleep_fn=lambda _: None,
        request_id_factory=lambda: "req-stable",
    )

    response = transport.send({"env": "toy"})

    assert response["request_id"] == "req-stable"
    assert [payload["request_id"] for payload in transport.seen_payloads] == [
        "req-stable",
        "req-stable",
    ]
