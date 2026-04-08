"""High-level API for one-call experiment execution.

This module provides a minimal configuration surface that maps to the existing
A/B testing architecture without removing advanced controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from .abtesting.models import ABTestConfig, StatisticalTestType, Variant
from .abtesting.runner import run_ab_test
from .deployment.platforms import get_platform


@dataclass(frozen=True)
class SimpleVariantConfig:
    """Minimal variant definition for high-level experiment runs."""

    model_path: str
    name: str | None = None
    traffic_weight: float = 1.0
    deployment_config: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunExperimentConfig:
    """Minimal schema for `run_experiment`.

    Fields are intentionally small; advanced controls remain available through
    passthrough dictionaries so no architecture features are lost.
    """

    variants: list[SimpleVariantConfig]
    primary_metric: str = "latency_ms_per_request"
    deployment_platform: str = "vllm"
    experiment_id: str | None = None
    name: str | None = None

    # Sane defaults
    secondary_metrics: list[str] = field(
        default_factory=lambda: ["throughput_toks_per_sec", "error_rate"]
    )
    traffic_allocation: float = 1.0
    significance_level: float = 0.05
    statistical_power: float = 0.8
    test_type: str = "t-test"
    min_sample_size_per_variant: int = 100
    max_duration_hours: float = 24.0

    # Hidden complexity knobs (optional)
    server_config: dict[str, Any] = field(default_factory=dict)
    rollout: dict[str, Any] = field(default_factory=dict)
    tokenizer_alignment: dict[str, Any] = field(default_factory=dict)

    # Full-fidelity escape hatches (preserves feature parity)
    health_checks: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def _normalize_test_type(test_type: str) -> StatisticalTestType:
    """Normalize user string to the existing statistical enum."""
    return StatisticalTestType[test_type.upper().replace("-", "_")]


def _normalize_variants(
    variants: list[SimpleVariantConfig],
    tokenizer_alignment: Mapping[str, Any],
) -> list[Variant]:
    """Convert high-level variant config to existing architecture model.

    Tokenizer alignment is stored in each variant's deployment config so lower
    layers can consume it without API callers wiring details manually.
    """
    tokenizer_policy = dict(tokenizer_alignment)
    normalized: list[Variant] = []

    for idx, variant in enumerate(variants):
        deployment_config = dict(variant.deployment_config or {})
        if tokenizer_policy:
            deployment_config.setdefault("tokenizer_alignment", tokenizer_policy)

        normalized.append(
            Variant(
                variant_id=f"variant-{idx}",
                name=variant.name or f"variant-{idx}",
                model_path=variant.model_path,
                deployment_config=deployment_config or None,
                traffic_weight=variant.traffic_weight,
                metadata=dict(variant.metadata),
            )
        )

    return normalized


def _coerce_config(config: RunExperimentConfig | Mapping[str, Any]) -> RunExperimentConfig:
    """Support dataclass config or dictionary input."""
    if isinstance(config, RunExperimentConfig):
        return config

    data = dict(config)
    raw_variants = data.get("variants", [])
    variants = [
        v if isinstance(v, SimpleVariantConfig) else SimpleVariantConfig(**v) for v in raw_variants
    ]
    return RunExperimentConfig(**{**data, "variants": variants})


def run_experiment(config: RunExperimentConfig | Mapping[str, Any]) -> Any:
    """Run a full experiment with a minimal API surface.

    This function hides:
    - deployment platform/server setup (via `get_platform`)
    - rollout handling (via `auto_stop_conditions`)
    - tokenizer alignment plumbing (in variant deployment config)

    It returns the existing `ExperimentResult` object from the lower-level
    experiment runner.
    """
    cfg = _coerce_config(config)

    if len(cfg.variants) < 2:
        raise ValueError("run_experiment requires at least two variants")

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_id = cfg.experiment_id or f"exp-{now}"
    experiment_name = cfg.name or f"experiment-{now}"

    variants = _normalize_variants(cfg.variants, cfg.tokenizer_alignment)

    auto_stop_conditions = dict(cfg.rollout)
    auto_stop_conditions.setdefault("monitoring_interval_seconds", 30.0)

    ab_config = ABTestConfig(
        experiment_id=experiment_id,
        name=experiment_name,
        variants=variants,
        primary_metric=cfg.primary_metric,
        secondary_metrics=list(cfg.secondary_metrics),
        traffic_allocation=cfg.traffic_allocation,
        significance_level=cfg.significance_level,
        statistical_power=cfg.statistical_power,
        test_type=_normalize_test_type(cfg.test_type),
        min_sample_size_per_variant=cfg.min_sample_size_per_variant,
        max_duration_hours=cfg.max_duration_hours,
        auto_stop_conditions=auto_stop_conditions,
        deployment_platform=cfg.deployment_platform,
        deployment_strategy="immediate",
        health_checks=dict(cfg.health_checks),
        metadata=dict(cfg.metadata),
    )

    platform = get_platform(cfg.deployment_platform, config=dict(cfg.server_config))
    return run_ab_test(ab_config, platform)
