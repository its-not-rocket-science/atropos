"""Transport client abstraction with retry policy hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransportClient:
    """Boundary for API/model communication.

    Subclasses can override ``_send_once`` for concrete HTTP/SDK integrations.
    """

    max_retries: int = 2
    retriable_exceptions: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))

    def _send_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "payload": payload}

    def send(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self._send_once(payload)
            except self.retriable_exceptions as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError("Transport retries exhausted") from exc
                attempt += 1
