"""Deployment strategies for advanced rollout patterns.

Provides canary, blue-green, rolling update, and A/B testing strategies for
controlled, low-risk deployments with health check integration.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import DeploymentRequest, DeploymentResult
    from .platforms import DeploymentPlatform


class DeploymentStrategy(ABC):
    """Abstract base class for deployment strategies."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize strategy with configuration.

        Args:
            config: Strategy-specific configuration.
        """
        self.config = config or {}

    @abstractmethod
    def execute(
        self,
        platform: DeploymentPlatform,
        request: DeploymentRequest,
    ) -> DeploymentResult:
        """Execute deployment using this strategy.

        Args:
            platform: Deployment platform to use.
            request: Deployment request.

        Returns:
            DeploymentResult with outcome.

        Raises:
            ValueError: If request or configuration is invalid.
            RuntimeError: If deployment fails.
        """
        raise NotImplementedError


class ImmediateStrategy(DeploymentStrategy):
    """Immediate deployment - full rollout at once."""

    def execute(
        self,
        platform: DeploymentPlatform,
        request: DeploymentRequest,
    ) -> DeploymentResult:
        """Execute immediate deployment."""
        # Simply delegate to platform's deploy method
        return platform.deploy(request)


class CanaryStrategy(DeploymentStrategy):
    """Canary deployment - gradual traffic shift with health checks."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.initial_percent = self.config.get("initial_percent", 10.0)
        self.increment_percent = self.config.get("increment_percent", 10.0)
        self.interval_minutes = self.config.get("interval_minutes", 5)
        self.health_check_timeout = self.config.get("health_check_timeout", 300)
        self.max_errors = self.config.get("max_errors", 5)

    def execute(
        self,
        platform: DeploymentPlatform,
        request: DeploymentRequest,
    ) -> DeploymentResult:
        """Execute canary deployment."""
        from .models import DeploymentResult, DeploymentStatus

        # Start deployment
        initial_result = platform.deploy(request)

        if initial_result.status != DeploymentStatus.SUCCESS:
            return initial_result

        deployment_id = initial_result.deployment_id
        if not deployment_id:
            return DeploymentResult(
                request=request,
                status=DeploymentStatus.FAILED,
                message="Platform did not return deployment ID",
                start_time=initial_result.start_time,
                end_time=initial_result.end_time,
            )

        # Simulate canary rollout (TODO: implement actual traffic shifting)
        # For now, just simulate health checks
        current_percent = self.initial_percent
        errors = 0

        while current_percent <= 100.0:
            # Check health
            health_result = platform.get_status(deployment_id)
            if health_result.status != DeploymentStatus.SUCCESS:
                errors += 1
                if errors >= self.max_errors:
                    # Rollback
                    rollback_result = platform.rollback(deployment_id)
                    return DeploymentResult(
                        request=request,
                        status=DeploymentStatus.FAILED,
                        message=(
                            f"Canary deployment failed at {current_percent}%: "
                            f"{rollback_result.message}"
                        ),
                        start_time=initial_result.start_time,
                        end_time=datetime.fromtimestamp(time.time()).isoformat(),
                        deployment_id=deployment_id,
                    )

            # Simulate traffic increase
            if current_percent < 100.0:
                time.sleep(self.interval_minutes * 60)  # Convert minutes to seconds
                current_percent = min(100.0, current_percent + self.increment_percent)
            else:
                break

        return DeploymentResult(
            request=request,
            status=DeploymentStatus.SUCCESS,
            message="Canary deployment completed successfully (100% traffic)",
            start_time=initial_result.start_time,
            end_time=datetime.fromtimestamp(time.time()).isoformat(),
            deployment_id=deployment_id,
            endpoints=initial_result.endpoints,
            metrics=initial_result.metrics,
        )


class BlueGreenStrategy(DeploymentStrategy):
    """Blue-green deployment - alternate environment swap."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.validation_duration = self.config.get("validation_duration", 15)  # minutes
        self.auto_swap = self.config.get("auto_swap", True)

    def execute(
        self,
        platform: DeploymentPlatform,
        request: DeploymentRequest,
    ) -> DeploymentResult:
        """Execute blue-green deployment."""
        from .models import DeploymentResult, DeploymentStatus

        # Deploy to green environment
        start_time = datetime.fromtimestamp(time.time()).isoformat()

        # Modify request for green deployment
        green_request = request
        green_request.metadata = green_request.metadata.copy()
        green_request.metadata["environment"] = "green"

        green_result = platform.deploy(green_request)
        if green_result.status != DeploymentStatus.SUCCESS:
            return green_result

        deployment_id = green_result.deployment_id
        if deployment_id is None:
            return DeploymentResult(
                request=request,
                status=DeploymentStatus.FAILED,
                message="Platform did not return deployment ID",
                start_time=start_time,
                end_time=datetime.fromtimestamp(time.time()).isoformat(),
            )

        # Validate green deployment
        validation_start = time.time()
        while time.time() - validation_start < self.validation_duration * 60:
            health_result = platform.get_status(deployment_id)
            if health_result.status != DeploymentStatus.SUCCESS:
                # Clean up green deployment
                platform.delete(deployment_id)
                return DeploymentResult(
                    request=request,
                    status=DeploymentStatus.FAILED,
                    message=f"Green deployment failed validation: {health_result.message}",
                    start_time=start_time,
                    end_time=datetime.fromtimestamp(time.time()).isoformat(),
                )
            time.sleep(30)  # Check every 30 seconds

        # TODO: Implement actual traffic swap
        # For now, simulate swap by updating endpoints

        return DeploymentResult(
            request=request,
            status=DeploymentStatus.SUCCESS,
            message="Blue-green deployment completed successfully",
            start_time=start_time,
            end_time=datetime.fromtimestamp(time.time()).isoformat(),
            deployment_id=deployment_id,
            endpoints=green_result.endpoints,
            metrics=green_result.metrics,
        )


class RollingUpdateStrategy(DeploymentStrategy):
    """Rolling update - incremental instance updates."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.batch_size = self.config.get("batch_size", 1)
        self.wait_between_batches = self.config.get("wait_between_batches", 60)  # seconds
        self.max_failed_batches = self.config.get("max_failed_batches", 1)

    def execute(
        self,
        platform: DeploymentPlatform,
        request: DeploymentRequest,
    ) -> DeploymentResult:
        """Execute rolling update deployment."""

        # TODO: Implement actual rolling update across instances
        # For now, simulate as immediate deployment
        return platform.deploy(request)


class ABTestStrategy(DeploymentStrategy):
    """A/B testing deployment with multiple variants."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.experiment_id = self.config.get("experiment_id")
        self.traffic_allocation = self.config.get("traffic_allocation", 1.0)
        self.auto_start = self.config.get("auto_start", True)

    def execute(
        self,
        platform: DeploymentPlatform,
        request: DeploymentRequest,
    ) -> DeploymentResult:
        """Execute A/B test deployment."""
        from dataclasses import replace

        from .models import DeploymentResult, DeploymentStatus

        start_time = datetime.fromtimestamp(time.time()).isoformat()

        # Extract experiment config from request
        experiment_config = request.experiment_config
        if not experiment_config:
            return DeploymentResult(
                request=request,
                status=DeploymentStatus.FAILED,
                message="No experiment configuration provided",
                start_time=start_time,
                end_time=datetime.fromtimestamp(time.time()).isoformat(),
            )

        # Parse experiment config (could be ABTestConfig dict or raw dict)
        try:
            # Try to parse as ABTestConfig
            if "experiment_id" in experiment_config and "variants" in experiment_config:
                # Already in ABTestConfig format
                config_dict = experiment_config
            else:
                # Convert from raw format
                config_dict = self._parse_experiment_config(experiment_config)

            # Deploy all variants for A/B testing
            if config_dict.get("variants"):
                variants = config_dict["variants"]
                experiment_id = config_dict.get("experiment_id", f"exp-{int(time.time())}")

                variant_deployments: dict[str, dict[str, Any]] = {}
                deployed_variants: list[dict[str, str | None]] = []
                all_endpoints: list[str] = []

                # Normalize traffic weights
                total_weight = sum(v.get("traffic_weight", 1.0) for v in variants)
                normalized_weights = {
                    v["variant_id"]: v.get("traffic_weight", 1.0) / total_weight for v in variants
                }

                # Deploy each variant
                for variant in variants:
                    variant_id = variant["variant_id"]
                    # Create deployment request for this variant
                    variant_request = replace(
                        request,
                        model_path=variant.get("model_path", request.model_path),
                        metadata={
                            **request.metadata,
                            "experiment_id": experiment_id,
                            "variant_id": variant_id,
                        },
                    )

                    # Deploy variant
                    variant_result = platform.deploy(variant_request)

                    if variant_result.status != DeploymentStatus.SUCCESS:
                        # Rollback previously deployed variants
                        for deployed_variant in deployed_variants:
                            if deployed_variant["deployment_id"] is not None:
                                platform.delete(deployed_variant["deployment_id"])
                        return variant_result

                    if variant_result.deployment_id is None:
                        # Rollback previously deployed variants
                        for deployed_variant in deployed_variants:
                            if deployed_variant["deployment_id"] is not None:
                                platform.delete(deployed_variant["deployment_id"])
                        return DeploymentResult(
                            request=request,
                            status=DeploymentStatus.FAILED,
                            message=(
                                f"Platform did not return deployment ID for variant {variant_id}"
                            ),
                            start_time=start_time,
                            end_time=datetime.fromtimestamp(time.time()).isoformat(),
                        )

                    # Store deployment info
                    variant_deployments[variant_id] = {
                        "deployment_id": variant_result.deployment_id,
                        "endpoints": variant_result.endpoints,
                        "weight": normalized_weights[variant_id],
                    }
                    deployed_variants.append(
                        {
                            "variant_id": variant_id,
                            "deployment_id": variant_result.deployment_id,
                        }
                    )
                    all_endpoints.extend(variant_result.endpoints)

                # Return success with experiment ID
                return DeploymentResult(
                    request=request,
                    status=DeploymentStatus.SUCCESS,
                    message=(
                        f"A/B test experiment '{experiment_id}' started "
                        f"with {len(variants)} variants"
                    ),
                    start_time=start_time,
                    end_time=datetime.fromtimestamp(time.time()).isoformat(),
                    deployment_id=experiment_id,
                    endpoints=all_endpoints[:10],  # Limit to first 10 endpoints
                    metrics={
                        "variant_count": len(variants),
                        "traffic_allocation": self.traffic_allocation,
                        "variant_deployments": variant_deployments,
                        "normalized_weights": normalized_weights,
                    },
                )
            else:
                return DeploymentResult(
                    request=request,
                    status=DeploymentStatus.FAILED,
                    message="No variants defined in experiment configuration",
                    start_time=start_time,
                    end_time=datetime.fromtimestamp(time.time()).isoformat(),
                )
        except Exception as e:
            return DeploymentResult(
                request=request,
                status=DeploymentStatus.FAILED,
                message=f"Failed to parse experiment configuration: {str(e)}",
                start_time=start_time,
                end_time=datetime.fromtimestamp(time.time()).isoformat(),
            )

    def _parse_experiment_config(self, raw_config: dict[str, Any]) -> dict[str, Any]:
        """Parse raw experiment configuration into standardized format."""
        # Simple implementation: assume raw_config is already in correct format
        # In real implementation, would validate and convert to ABTestConfig
        return raw_config


# Registry of available strategies
STRATEGIES: dict[str, type[DeploymentStrategy]] = {
    "immediate": ImmediateStrategy,
    "canary": CanaryStrategy,
    "blue-green": BlueGreenStrategy,
    "rolling": RollingUpdateStrategy,
    "a-b-test": ABTestStrategy,
}


def get_strategy(
    strategy_name: str,
    config: dict[str, Any] | None = None,
) -> DeploymentStrategy:
    """Get a deployment strategy by name.

    Args:
        strategy_name: One of 'immediate', 'canary', 'blue-green', 'rolling'.
        config: Strategy-specific configuration.

    Returns:
        DeploymentStrategy instance.

    Raises:
        ValueError: If strategy is not recognized.
    """
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. Available: {list(STRATEGIES.keys())}"
        )

    strategy_class = STRATEGIES[strategy_name]
    return strategy_class(config)
