"""Transport client abstraction with request IDs, retries, and error taxonomy."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from time import sleep
from typing import Any
from uuid import uuid4


class TransportError(RuntimeError):
    """Base class for transport errors."""


class RetryableTransportError(TransportError):
    """Transport error that can be retried safely."""


class NonRetryableTransportError(TransportError):
    """Transport error that should fail fast."""


class TransportTimeoutError(RetryableTransportError):
    """Timeout while attempting to send transport payload."""


class TransportServerError(RetryableTransportError):
    """Server-side (5xx) transport failure."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Server error ({status_code}): {message}")
        self.status_code = status_code


class TransportClientError(NonRetryableTransportError):
    """Client-side (4xx) transport failure."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(f"Client error ({status_code}): {message}")
        self.status_code = status_code


class TransportMalformedResponseError(NonRetryableTransportError):
    """Transport returned invalid response shape."""


class TransportRetriesExhaustedError(RetryableTransportError):
    """Retry budget consumed without a successful response."""

    def __init__(self, request_id: str, retry_count: int, last_error: Exception) -> None:
        super().__init__(
            f"Transport retries exhausted for request_id={request_id} after {retry_count} retries"
        )
        self.request_id = request_id
        self.retry_count = retry_count
        self.last_error = last_error


@dataclass
class TransportClient:
    """Boundary for API/model communication.

    Subclasses can override ``_send_once`` for concrete HTTP/SDK integrations.
    """

    max_retries: int = 2
    retriable_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (RuntimeError,)
    )
    base_backoff_seconds: float = 0.05
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 1.0
    sleep_fn: Callable[[float], None] = sleep
    request_id_factory: Callable[[], str] = field(default_factory=lambda: lambda: str(uuid4()))
    metrics: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def _send_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "payload": payload}

    def _compute_backoff(self, retry_count: int) -> float:
        backoff = self.base_backoff_seconds * (self.backoff_multiplier ** (retry_count - 1))
        return min(backoff, self.max_backoff_seconds)

    def _classify_exception(self, exc: Exception) -> TransportError:
        if isinstance(exc, TransportError):
            return exc
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            if 500 <= status_code < 600:
                return TransportServerError(status_code, str(exc))
            if 400 <= status_code < 500:
                return TransportClientError(status_code, str(exc))
        if isinstance(exc, TimeoutError):
            return TransportTimeoutError(str(exc))
        if isinstance(exc, self.retriable_exceptions):
            return RetryableTransportError(str(exc))
        return NonRetryableTransportError(str(exc))

    def _normalize_response(self, response: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(response, dict):
            raise TransportMalformedResponseError("Transport response must be a dictionary")

        status_code = response.get("status_code")
        if isinstance(status_code, int):
            message = str(response.get("error", "transport status response"))
            if 500 <= status_code < 600:
                raise TransportServerError(status_code, message)
            if 400 <= status_code < 500:
                raise TransportClientError(status_code, message)

        ok = response.get("ok")
        if not isinstance(ok, bool):
            raise TransportMalformedResponseError("Transport response must include boolean 'ok'")

        return dict(response)

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        outbound_payload = dict(payload)
        original_has_request_id = "request_id" in outbound_payload
        request_id = str(outbound_payload.get("request_id") or self.request_id_factory())
        outbound_payload["request_id"] = request_id

        retry_count = 0
        last_error: TransportError | None = None
        while True:
            try:
                response = self._send_once(outbound_payload)
                normalized = self._normalize_response(response)
                if retry_count > 0:
                    normalized.setdefault("request_id", request_id)
                    normalized.setdefault("retry_count", retry_count)
                elif original_has_request_id:
                    normalized.setdefault("request_id", request_id)

                response_payload = normalized.get("payload")
                if (
                    retry_count == 0
                    and not original_has_request_id
                    and isinstance(response_payload, dict)
                    and response_payload.get("request_id") == request_id
                ):
                    response_payload = dict(response_payload)
                    response_payload.pop("request_id", None)
                    normalized["payload"] = response_payload
                self.metrics["transport_success_total"] += 1
                self.metrics["transport_retry_count_total"] += retry_count
                return normalized
            except Exception as raw_exc:
                error = self._classify_exception(raw_exc)
                last_error = error
                error_name = type(error).__name__
                self.metrics[f"transport_error_total:{error_name}"] += 1

                should_retry = (
                    isinstance(error, RetryableTransportError) and retry_count < self.max_retries
                )
                if not should_retry:
                    self.metrics["transport_terminal_failures_total"] += 1
                    self.metrics[f"transport_terminal_failures_total:{error_name}"] += 1
                    if isinstance(error, RetryableTransportError):
                        raise TransportRetriesExhaustedError(
                            request_id=request_id,
                            retry_count=retry_count,
                            last_error=last_error,
                        ) from raw_exc
                    raise error from raw_exc

                retry_count += 1
                self.metrics["transport_retries_total"] += 1
                self.metrics[f"transport_retries_total:{error_name}"] += 1
                backoff_seconds = self._compute_backoff(retry_count)
                self.sleep_fn(backoff_seconds)
