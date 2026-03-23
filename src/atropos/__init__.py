"""Atropos: ROI estimation for LLM pruning and related optimizations."""

from .calculations import combine_strategies, estimate_outcome
from .models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy

__all__ = [
    "DeploymentScenario",
    "OptimizationOutcome",
    "OptimizationStrategy",
    "combine_strategies",
    "estimate_outcome",
]
__version__ = "0.4.0"
