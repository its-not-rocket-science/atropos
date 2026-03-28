"""Health check integration for deployment validation."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any


class HealthCheck(ABC):
    """Abstract base class for health checks."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize health check with configuration.

        Args:
            config: Health check configuration.
        """
        self.config = config or {}

    @abstractmethod
    def check(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Perform health check on an endpoint.

        Args:
            endpoint: The endpoint to check (URL, path, etc.).
            **kwargs: Additional check-specific parameters.

        Returns:
            Dictionary with check results including:
            - 'healthy': bool
            - 'latency_ms': float (optional)
            - 'error': str (optional)
            - Additional check-specific metrics.
        """
        raise NotImplementedError


class HttpHealthCheck(HealthCheck):
    """HTTP endpoint health check."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.timeout = self.config.get("timeout", 5)
        self.expected_status = self.config.get("expected_status", 200)
        self.expected_text = self.config.get("expected_text")

    def check(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Check HTTP endpoint health."""
        start_time = time.time()
        try:
            req = urllib.request.Request(endpoint)
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                latency_ms = (time.time() - start_time) * 1000
                status = response.getcode()
                body = response.read().decode("utf-8") if self.expected_text else ""

                healthy = status == self.expected_status
                if self.expected_text and healthy:
                    healthy = self.expected_text in body

                result = {
                    "healthy": healthy,
                    "latency_ms": latency_ms,
                    "status_code": status,
                    "endpoint": endpoint,
                }

                if not healthy:
                    result["error"] = f"HTTP {status} (expected {self.expected_status})"
                    if self.expected_text and self.expected_text not in body:
                        result["error"] = "Expected text not found in response"

                return result

        except urllib.error.URLError as e:
            return {
                "healthy": False,
                "latency_ms": (time.time() - start_time) * 1000,
                "error": str(e),
                "endpoint": endpoint,
            }
        except Exception as e:
            return {
                "healthy": False,
                "latency_ms": (time.time() - start_time) * 1000,
                "error": f"Unexpected error: {e}",
                "endpoint": endpoint,
            }


class LatencyHealthCheck(HealthCheck):
    """Latency threshold health check."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.max_latency_ms = self.config.get("max_latency_ms", 1000)
        self.delegate = HttpHealthCheck(config)  # Use HTTP check by default

    def check(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Check endpoint latency against threshold."""
        result = self.delegate.check(endpoint, **kwargs)

        if "latency_ms" in result:
            result["latency_healthy"] = result["latency_ms"] <= self.max_latency_ms
            if not result["latency_healthy"]:
                result["error"] = (
                    f"Latency {result['latency_ms']:.1f}ms exceeds "
                    f"threshold {self.max_latency_ms}ms"
                )
                result["healthy"] = False

        return result


class JsonHealthCheck(HealthCheck):
    """JSON response health check with field validation."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.expected_fields = self.config.get("expected_fields", {})
        self.delegate = HttpHealthCheck(config)

    def check(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Check JSON endpoint with field validation."""
        result = self.delegate.check(endpoint, **kwargs)

        if not result.get("healthy", False):
            return result

        # Try to parse JSON
        try:
            # Get response body from delegate (would need to store it)
            # For simplicity, make another request
            req = urllib.request.Request(endpoint)
            with urllib.request.urlopen(req, timeout=self.config.get("timeout", 5)) as resp:
                body = resp.read().decode("utf-8")
                data = json.loads(body)

                # Validate expected fields
                for field, expected_value in self.expected_fields.items():
                    if field not in data:
                        result["healthy"] = False
                        result["error"] = f"Missing field: {field}"
                        break
                    if data[field] != expected_value:
                        result["healthy"] = False
                        result["error"] = (
                            f"Field {field} value mismatch: {data[field]} != {expected_value}"
                        )
                        break

                result["json_data"] = data

        except json.JSONDecodeError as e:
            result["healthy"] = False
            result["error"] = f"Invalid JSON: {e}"
        except Exception as e:
            result["healthy"] = False
            result["error"] = f"JSON check error: {e}"

        return result


class CompositeHealthCheck(HealthCheck):
    """Composite health check that runs multiple checks."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.checks = self.config.get("checks", [])
        self.require_all = self.config.get("require_all", True)

    def check(self, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        """Run all configured health checks."""
        results = []
        all_healthy = True
        any_healthy = False

        for check_config in self.checks:
            check_type = check_config.get("type", "http")
            check = self._create_check(check_type, check_config)
            result = check.check(endpoint, **kwargs)
            results.append(result)

            if result.get("healthy", False):
                any_healthy = True
            else:
                all_healthy = False

        overall_healthy = all_healthy if self.require_all else any_healthy

        return {
            "healthy": overall_healthy,
            "checks": results,
            "endpoint": endpoint,
        }

    def _create_check(self, check_type: str, config: dict[str, Any]) -> HealthCheck:
        """Create a health check instance by type."""
        if check_type == "http":
            return HttpHealthCheck(config)
        if check_type == "latency":
            return LatencyHealthCheck(config)
        if check_type == "json":
            return JsonHealthCheck(config)
        raise ValueError(f"Unknown health check type: {check_type}")


# Registry of available health check types
HEALTH_CHECKS: dict[str, type[HealthCheck]] = {
    "http": HttpHealthCheck,
    "latency": LatencyHealthCheck,
    "json": JsonHealthCheck,
    "composite": CompositeHealthCheck,
}


def get_health_check(
    check_type: str,
    config: dict[str, Any] | None = None,
) -> HealthCheck:
    """Get a health check by type.

    Args:
        check_type: One of 'http', 'latency', 'json', 'composite'.
        config: Health check configuration.

    Returns:
        HealthCheck instance.

    Raises:
        ValueError: If check_type is not recognized.
    """
    if check_type not in HEALTH_CHECKS:
        raise ValueError(
            f"Unknown health check type '{check_type}'. Available: {list(HEALTH_CHECKS.keys())}"
        )

    check_class = HEALTH_CHECKS[check_type]
    return check_class(config)
