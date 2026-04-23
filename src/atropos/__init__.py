"""Atropos: ROI estimation for LLM pruning and related optimizations.

Stability tier: mixed package. Tier 2 (supported research infrastructure)
for most `atropos.*` modules; see docs/stability-tiers.md for exceptions.
"""

from .api import RunExperimentConfig, SimpleVariantConfig, run_experiment
from .calculations import combine_strategies, estimate_outcome
from .models import DeploymentScenario, OptimizationOutcome, OptimizationStrategy

__all__ = [
    "DeploymentScenario",
    "OptimizationOutcome",
    "OptimizationStrategy",
    "combine_strategies",
    "estimate_outcome",
    "SimpleVariantConfig",
    "RunExperimentConfig",
    "run_experiment",
]
__version__ = "0.5.0"
