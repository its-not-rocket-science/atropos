"""Deployment automation for production deployment platforms.

Provides integration with vLLM, Triton, SageMaker, and other deployment platforms
with support for advanced deployment strategies (canary, blue-green, rolling updates)
and health check integration.
"""

from __future__ import annotations

__all__ = [
    "DeploymentPlatform",
    "DeploymentStrategy",
    "DeploymentResult",
    "HealthCheck",
    "get_platform",
    "get_strategy",
    "get_health_check",
    "PLATFORMS",
    "STRATEGIES",
    "HEALTH_CHECKS",
]

from .health import HEALTH_CHECKS, HealthCheck, get_health_check
from .models import DeploymentResult
from .platforms import PLATFORMS, DeploymentPlatform, get_platform
from .strategies import STRATEGIES, DeploymentStrategy, get_strategy
