"""Hyperparameter tuning for optimization strategies.

This module provides automated tuning of pruning strategy parameters
based on model characteristics and deployment constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..logging_config import get_logger
from ..models import DeploymentScenario, OptimizationStrategy, QualityRisk
from ..presets import QUANTIZATION_BONUS, STRATEGIES
from ..pruning_integration import auto_select_framework

logger = get_logger("tuning")


class ModelArchitecture(Enum):
    """Common LLM architecture families."""

    GPT2 = "gpt2"
    GPT_NEO = "gpt_neo"
    GPT_J = "gpt_j"
    LLAMA = "llama"
    BLOOM = "bloom"
    OPT = "opt"
    PYTHIA = "pythia"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    UNKNOWN = "unknown"


@dataclass
class ModelCharacteristics:
    """Characteristics of a model for tuning."""

    parameter_count_b: float
    architecture: ModelArchitecture
    layer_types: list[str]
    attention_heads: int | None = None
    hidden_size: int | None = None
    num_layers: int | None = None
    has_sparse_support: bool = False


@dataclass
class TuningConstraints:
    """Deployment constraints for tuning optimization."""

    max_memory_gb: float | None = None
    min_throughput_toks_per_sec: float | None = None
    max_latency_ms_per_request: float | None = None
    max_power_watts: float | None = None
    max_quality_risk: QualityRisk = "medium"
    use_quantization: bool = False
    prefer_fast_pruning: bool = False


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    strategy: OptimizationStrategy
    recommended_framework: str
    quality_risk: QualityRisk
    expected_memory_gb: float
    expected_throughput_toks_per_sec: float
    expected_latency_ms_per_request: float
    expected_power_watts: float
    tuning_metadata: dict[str, Any] = field(default_factory=dict)


class HyperparameterTuner:
    """Tunes pruning strategy parameters for optimal ROI.

    Analyzes model architecture and deployment constraints to recommend
    optimized pruning strategy parameters.
    """

    def __init__(
        self,
        scenario: DeploymentScenario,
        model_characteristics: ModelCharacteristics | None = None,
        constraints: TuningConstraints | None = None,
    ):
        """Initialize tuner with scenario and optional constraints.

        Args:
            scenario: Deployment scenario.
            model_characteristics: Model characteristics (inferred if None).
            constraints: Deployment constraints (optional).
        """
        self.scenario = scenario
        self.constraints = constraints or TuningConstraints()

        # Infer model characteristics if not provided
        self.model_chars = model_characteristics or self._infer_model_characteristics()

        # Framework recommendation cache
        self._recommended_framework: str | None = None

        logger.debug(
            "Initialized tuner: scenario=%s, architecture=%s, params=%.1fB",
            scenario.name,
            self.model_chars.architecture.value,
            self.model_chars.parameter_count_b,
        )

    def _infer_model_characteristics(self) -> ModelCharacteristics:
        """Infer model characteristics from scenario parameters."""
        params_b = self.scenario.parameters_b

        # Simple architecture inference based on parameter count
        # This could be enhanced with actual model loading
        if params_b <= 1.5:
            architecture = ModelArchitecture.GPT2
        elif params_b <= 13:
            architecture = ModelArchitecture.LLAMA
        else:
            architecture = ModelArchitecture.UNKNOWN

        # Default layer types for transformer models
        layer_types = ["attention", "mlp", "layer_norm", "embedding", "lm_head"]

        # Estimate attention heads and hidden size
        # Rough heuristic: sqrt(parameters) ~ hidden_size

        return ModelCharacteristics(
            parameter_count_b=params_b,
            architecture=architecture,
            layer_types=layer_types,
            has_sparse_support=False,  # Assume no sparse support by default
        )

    def tune(self) -> TuningResult:
        """Run hyperparameter tuning.

        Returns:
            TuningResult with optimized strategy and recommendations.
        """
        logger.info("Starting hyperparameter tuning for scenario: %s", self.scenario.name)

        # Step 1: Analyze model and constraints
        logger.debug("Analyzing model characteristics...")
        model_analysis = self._analyze_model()

        # Step 2: Determine optimization objectives
        logger.debug("Determining optimization objectives...")
        objectives = self._determine_objectives(model_analysis)

        # Step 3: Optimize strategy parameters
        logger.debug("Optimizing strategy parameters...")
        strategy = self._optimize_parameters(objectives)

        # Step 4: Recommend framework
        logger.debug("Recommending pruning framework...")
        framework = self._recommend_framework(strategy)

        # Step 5: Apply quantization if requested
        if self.constraints.use_quantization:
            logger.debug("Applying quantization bonus...")
            from ..calculations import combine_strategies

            strategy = combine_strategies(strategy, QUANTIZATION_BONUS)

        # Step 6: Estimate expected metrics
        logger.debug("Estimating expected metrics...")
        expected_metrics = self._estimate_metrics(strategy)

        logger.info("Tuning complete. Recommended strategy: %s", strategy.name)

        return TuningResult(
            strategy=strategy,
            recommended_framework=framework,
            quality_risk=strategy.quality_risk,
            expected_memory_gb=expected_metrics["memory_gb"],
            expected_throughput_toks_per_sec=expected_metrics["throughput_toks_per_sec"],
            expected_latency_ms_per_request=expected_metrics["latency_ms_per_request"],
            expected_power_watts=expected_metrics["power_watts"],
            tuning_metadata={
                "model_analysis": model_analysis,
                "objectives": objectives,
                "architecture": self.model_chars.architecture.value,
            },
        )

    def _analyze_model(self) -> dict[str, Any]:
        """Analyze model for tuning."""
        chars = self.model_chars

        # Determine if model is suitable for aggressive pruning
        # Larger models with more parameters can tolerate more pruning
        suitability_score = min(1.0, chars.parameter_count_b / 10.0)

        # Check for known architectures with good pruning results
        known_good_architectures = {
            ModelArchitecture.GPT2,
            ModelArchitecture.OPT,
            ModelArchitecture.LLAMA,
        }

        is_known_architecture = chars.architecture in known_good_architectures

        return {
            "parameter_count_b": chars.parameter_count_b,
            "architecture": chars.architecture.value,
            "suitability_score": suitability_score,
            "is_known_architecture": is_known_architecture,
            "has_sparse_support": chars.has_sparse_support,
            "layer_count_estimate": int(chars.parameter_count_b * 0.5),  # Rough estimate
        }

    def _determine_objectives(self, model_analysis: dict[str, Any]) -> dict[str, float]:
        """Determine optimization objectives based on constraints and model."""
        objectives = {}

        # Memory reduction objective
        if self.constraints.max_memory_gb is not None:
            current_memory = self.scenario.memory_gb
            target_memory = self.constraints.max_memory_gb
            if target_memory < current_memory:
                memory_reduction_target = (current_memory - target_memory) / current_memory
                objectives["memory_reduction"] = min(memory_reduction_target, 0.5)  # Cap at 50%

        # Throughput improvement objective
        if self.constraints.min_throughput_toks_per_sec is not None:
            current_throughput = self.scenario.throughput_toks_per_sec
            target_throughput = self.constraints.min_throughput_toks_per_sec
            if target_throughput > current_throughput:
                throughput_improvement_target = (
                    target_throughput - current_throughput
                ) / current_throughput
                objectives["throughput_improvement"] = min(
                    throughput_improvement_target, 1.0
                )  # Cap at 100%

        # Latency reduction objective (inverse of throughput)
        if self.constraints.max_latency_ms_per_request is not None:
            current_latency = (
                self.scenario.tokens_per_request / self.scenario.throughput_toks_per_sec * 1000
            )
            target_latency = self.constraints.max_latency_ms_per_request
            if target_latency < current_latency:
                latency_reduction_target = (current_latency - target_latency) / current_latency
                objectives["latency_reduction"] = min(latency_reduction_target, 0.5)

        # Power reduction objective
        if self.constraints.max_power_watts is not None:
            current_power = self.scenario.power_watts
            target_power = self.constraints.max_power_watts
            if target_power < current_power:
                power_reduction_target = (current_power - target_power) / current_power
                objectives["power_reduction"] = min(power_reduction_target, 0.3)  # Cap at 30%

        # If no specific constraints, use balanced objectives
        if not objectives:
            model_suitability = model_analysis["suitability_score"]

            # Models with more parameters can tolerate more aggressive pruning
            objectives["memory_reduction"] = 0.15 * model_suitability
            objectives["throughput_improvement"] = 0.1 * model_suitability
            objectives["power_reduction"] = 0.08 * model_suitability

        logger.debug("Determined objectives: %s", objectives)
        return objectives

    def _optimize_parameters(self, objectives: dict[str, float]) -> OptimizationStrategy:
        """Optimize strategy parameters to meet objectives.

        Uses Bayesian optimization if scikit-optimize is available,
        otherwise falls back to heuristic rules.
        """
        import importlib.util

        use_bayesian = importlib.util.find_spec("skopt") is not None
        if use_bayesian:
            logger.debug("Using Bayesian optimization (scikit-optimize available)")
        else:
            logger.debug("Using heuristic optimization (scikit-optimize not available)")

        if use_bayesian:
            return self._optimize_with_bayesian(objectives)
        else:
            return self._optimize_with_heuristics(objectives)

    def _optimize_with_bayesian(self, objectives: dict[str, float]) -> OptimizationStrategy:
        """Optimize using Bayesian optimization."""
        import skopt

        # Define parameter space
        space = [
            skopt.space.Real(0.05, 0.5, name="parameter_reduction"),  # 5% to 50%
            skopt.space.Real(0.03, 0.4, name="memory_reduction"),  # 3% to 40%
            skopt.space.Real(0.02, 0.4, name="throughput_improvement"),  # 2% to 40%
            skopt.space.Real(0.01, 0.25, name="power_reduction"),  # 1% to 25%
        ]

        # Define objective function
        def objective(params: tuple[float, float, float, float]) -> float:
            param_red, mem_red, throughput_imp, power_red = params

            # Calculate quality risk based on parameters
            risk_score = self._calculate_quality_risk(param_red, mem_red, throughput_imp)

            # Calculate how well we meet objectives
            objective_score = 0.0
            weights = {
                "memory_reduction": 0.4,
                "throughput_improvement": 0.3,
                "power_reduction": 0.2,
                "risk": 0.1,
            }

            if "memory_reduction" in objectives:
                diff = abs(mem_red - objectives["memory_reduction"])
                objective_score -= weights["memory_reduction"] * diff

            if "throughput_improvement" in objectives:
                diff = abs(throughput_imp - objectives["throughput_improvement"])
                objective_score -= weights["throughput_improvement"] * diff

            if "power_reduction" in objectives:
                diff = abs(power_red - objectives["power_reduction"])
                objective_score -= weights["power_reduction"] * diff

            # Penalize high risk
            objective_score -= weights["risk"] * risk_score

            return objective_score

        # Run optimization
        n_calls = 20  # Reasonable default
        result = skopt.gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=False,
        )

        # Extract best parameters
        param_red, mem_red, throughput_imp, power_red = result.x

        # Determine quality risk
        quality_risk = self._determine_risk_level(param_red, mem_red, throughput_imp)

        # Create strategy name
        strategy_name = f"tuned_{self.scenario.name}_{int(param_red * 100)}p"

        return OptimizationStrategy(
            name=strategy_name,
            parameter_reduction_fraction=float(param_red),
            memory_reduction_fraction=float(mem_red),
            throughput_improvement_fraction=float(throughput_imp),
            power_reduction_fraction=float(power_red),
            quality_risk=quality_risk,
        )

    def _optimize_with_heuristics(self, objectives: dict[str, float]) -> OptimizationStrategy:
        """Optimize using heuristic rules."""
        # Start with baseline from existing strategies
        baseline = STRATEGIES["structured_pruning"]

        # Adjust based on objectives
        param_red = baseline.parameter_reduction_fraction
        mem_red = baseline.memory_reduction_fraction
        throughput_imp = baseline.throughput_improvement_fraction
        power_red = baseline.power_reduction_fraction

        # Adjust for memory objective
        if "memory_reduction" in objectives:
            target_mem = objectives["memory_reduction"]
            # Scale parameter reduction proportionally to memory target
            scale_factor = target_mem / 0.22  # structured_pruning baseline
            param_red = baseline.parameter_reduction_fraction * scale_factor
            mem_red = target_mem

        # Adjust for throughput objective
        if "throughput_improvement" in objectives:
            target_throughput = objectives["throughput_improvement"]
            scale_factor = target_throughput / 0.20  # structured_pruning baseline
            throughput_imp = target_throughput

        # Adjust for power objective
        if "power_reduction" in objectives:
            target_power = objectives["power_reduction"]
            scale_factor = target_power / 0.14  # structured_pruning baseline
            power_red = target_power

        # Apply caps to prevent unrealistic values
        param_red = min(max(param_red, 0.05), 0.5)
        mem_red = min(max(mem_red, 0.03), 0.4)
        throughput_imp = min(max(throughput_imp, 0.02), 0.4)
        power_red = min(max(power_red, 0.01), 0.25)

        # Determine quality risk
        quality_risk = self._determine_risk_level(param_red, mem_red, throughput_imp)

        # Create strategy name
        strategy_name = f"tuned_{self.scenario.name}_{int(param_red * 100)}p"

        return OptimizationStrategy(
            name=strategy_name,
            parameter_reduction_fraction=param_red,
            memory_reduction_fraction=mem_red,
            throughput_improvement_fraction=throughput_imp,
            power_reduction_fraction=power_red,
            quality_risk=quality_risk,
        )

    def _calculate_quality_risk(
        self,
        parameter_reduction: float,
        memory_reduction: float,
        throughput_improvement: float,
    ) -> float:
        """Calculate quality risk score (0-1, higher = more risk)."""
        # Base risk from parameter reduction
        risk = parameter_reduction * 0.6

        # Additional risk if memory reduction lags parameter reduction
        # (suggests unstructured pruning which is riskier)
        if memory_reduction < parameter_reduction * 0.7:
            risk += 0.2

        # High throughput improvement can indicate aggressive pruning
        if throughput_improvement > 0.3:
            risk += 0.1

        return min(risk, 1.0)

    def _determine_risk_level(
        self,
        parameter_reduction: float,
        memory_reduction: float,
        throughput_improvement: float,
    ) -> QualityRisk:
        """Determine quality risk level from parameters."""
        risk_score = self._calculate_quality_risk(
            parameter_reduction, memory_reduction, throughput_improvement
        )

        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "medium"
        else:
            return "high"

    def _recommend_framework(self, strategy: OptimizationStrategy) -> str:
        """Recommend pruning framework for the strategy."""
        if self._recommended_framework is None:
            # Use auto_select_framework as baseline
            baseline_framework = auto_select_framework(self.scenario, strategy)

            # Enhance with tuning considerations
            framework = self._enhance_framework_recommendation(baseline_framework, strategy)
            self._recommended_framework = framework

        return self._recommended_framework

    def _enhance_framework_recommendation(
        self,
        baseline_framework: str,
        strategy: OptimizationStrategy,
    ) -> str:
        """Enhance framework recommendation with tuning considerations."""
        framework = baseline_framework

        # Consider model architecture
        architecture = self.model_chars.architecture

        # Architecture-specific recommendations
        if architecture == ModelArchitecture.GPT2:
            # GPT2 works well with SparseGPT
            if strategy.parameter_reduction_fraction >= 0.2:
                framework = "sparsegpt"
        elif architecture == ModelArchitecture.LLAMA:
            # Llama models often use Wanda or patched versions
            if self.model_chars.has_sparse_support:
                framework = "wanda-patched"
            else:
                framework = "wanda"
        elif architecture == ModelArchitecture.OPT:
            # OPT works well with magnitude pruning
            framework = "magnitude_pruning"

        # Consider speed preference
        if self.constraints.prefer_fast_pruning:
            fast_frameworks = {"wanda", "sparsegpt", "wanda-patched"}
            if framework not in fast_frameworks:
                # Choose fastest available
                framework = "wanda"

        logger.debug("Recommended framework: %s (baseline: %s)", framework, baseline_framework)
        return framework

    def _estimate_metrics(self, strategy: OptimizationStrategy) -> dict[str, float]:
        """Estimate expected metrics from optimized strategy."""
        # Apply strategy to scenario
        memory_gb = self.scenario.memory_gb * (1 - strategy.memory_reduction_fraction)
        throughput = self.scenario.throughput_toks_per_sec * (
            1 + strategy.throughput_improvement_fraction
        )
        power_watts = self.scenario.power_watts * (1 - strategy.power_reduction_fraction)

        # Calculate latency
        latency_ms = (self.scenario.tokens_per_request / throughput) * 1000

        return {
            "memory_gb": memory_gb,
            "throughput_toks_per_sec": throughput,
            "latency_ms_per_request": latency_ms,
            "power_watts": power_watts,
        }
