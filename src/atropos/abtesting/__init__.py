"""A/B testing framework for model variants.

Provides statistical comparison of model variants (pruned vs original,
different optimization strategies) with automatic traffic routing,
metric collection, significance testing, and promotion of winning variants.
"""

from __future__ import annotations

__all__ = [
    "ABTestConfig",
    "Variant",
    "ExperimentResult",
    "StatisticalResult",
    "ExperimentStatus",
    "StatisticalTestType",
    "VariantMetrics",
    "independent_t_test",
    "confidence_interval",
    "effect_size_cohens_d",
    "statistical_power",
    "sample_size_for_power",
    "mann_whitney_u_test",
    "analyze_variant_comparison",
    "SCIPY_AVAILABLE",
    "run_ab_test",
    "analyze_experiment_results",
]

# Import statements will be added as modules are implemented
from .models import (
    ABTestConfig,
    ExperimentResult,
    ExperimentStatus,
    StatisticalResult,
    StatisticalTestType,
    Variant,
    VariantMetrics,
)
from .runner import analyze_experiment_results, run_ab_test
from .statistics import (
    SCIPY_AVAILABLE,
    analyze_variant_comparison,
    confidence_interval,
    effect_size_cohens_d,
    independent_t_test,
    mann_whitney_u_test,
    sample_size_for_power,
    statistical_power,
)
