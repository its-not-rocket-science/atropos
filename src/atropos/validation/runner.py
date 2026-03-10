"""Validation runner for testing Atropos against real models."""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any, cast

import torch
from torch.nn.utils import prune
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from ..calculations import estimate_outcome
from .models import ComparisonMetric, MeasuredMetrics, ValidationResult

if TYPE_CHECKING:
    from ..models import DeploymentScenario, OptimizationStrategy



def _compute_variance(projected: float, measured: float) -> float:
    """Compute percentage variance between projected and measured."""
    if projected == 0:
        return float("inf") if measured != 0 else 0.0
    return ((measured - projected) / projected) * 100


class ModelValidator:
    """Validator for testing Atropos projections against real models."""

    def __init__(
        self,
        scenario: DeploymentScenario,
        strategy: OptimizationStrategy,
        device: str = "cpu",
    ):
        """Initialize validator.

        Args:
            scenario: Deployment scenario to validate.
            strategy: Optimization strategy to apply.
            device: Device to run on ("cpu" or "cuda").
        """
        self.scenario = scenario
        self.strategy = strategy
        self.device = device
        self._atropos_outcome = estimate_outcome(scenario, strategy)

    def measure_baseline(self, model_name: str | None = None) -> MeasuredMetrics:
        """Measure baseline (unoptimized) model metrics.

        This method attempts to load and measure an actual model.
        Falls back to simulation if model not available.

        Args:
            model_name: HuggingFace model name or path.

        Returns:
            MeasuredMetrics for baseline model.
        """
        model_name = model_name or self._infer_model_name()

        try:
            return self._measure_real_model(model_name, optimized=False)
        except ImportErrorException:
            # Fall back to simulated measurements
            return self._simulate_baseline_measurement(model_name)

    def measure_optimized(
        self,
        model_name: str | None = None,
        pruning_method: str = "magnitude",
    ) -> MeasuredMetrics:
        """Measure optimized (pruned) model metrics.

        Args:
            model_name: HuggingFace model name or path.
            pruning_method: Pruning method to apply.

        Returns:
            MeasuredMetrics for optimized model.
        """
        model_name = model_name or self._infer_model_name()

        try:
            return self._measure_real_model(
                model_name, optimized=True, pruning_method=pruning_method
            )
        except ImportErrorException:
            # Fall back to simulated measurements
            return self._simulate_optimized_measurement(model_name)

    def _infer_model_name(self) -> str:
        """Infer appropriate model name from scenario parameters."""
        params_b = self.scenario.parameters_b

        # Map parameter ranges to suitable test models
        if params_b <= 0.5:
            return "gpt2"  # 124M params
        elif params_b <= 1.0:
            return "gpt2-medium"  # 355M params
        elif params_b <= 3.0:
            return "gpt2-large"  # 774M params
        elif params_b <= 7.0:
            return "gpt2-xl"  # 1.5B params
        else:
            return "meta-llama/Llama-2-7b-hf"  # Fallback to requesting Llama

    def _measure_real_model(
        self,
        model_name: str,
        optimized: bool = False,
        pruning_method: str = "magnitude",
    ) -> MeasuredMetrics:
        """Measure a real PyTorch/HuggingFace model.

        Args:
            model_name: Model identifier.
            optimized: Whether to apply pruning.
            pruning_method: Pruning method.

        Returns:
            MeasuredMetrics from actual execution.

        Raises:
            ImportErrorException: If required packages not installed.
        """
        
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        loaded_model = AutoModelForCausalLM.from_pretrained(model_name)
        model = cast(PreTrainedModel, loaded_model)
        model = model.to(self.device)  # type: ignore[arg-type]
        model.eval()

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        parameters_b = param_count / 1e9

        # Apply pruning if optimized
        if optimized:
            model = self._apply_pruning(model, pruning_method)
            # Recount after pruning
            param_count = sum(p.numel() for p in model.parameters())
            parameters_b = param_count / 1e9

        # Measure memory
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Prepare test input
        test_text = "The quick brown fox jumps over the lazy dog. " * 10
        inputs = tokenizer(test_text, return_tensors="pt").to(self.device)

        # Warmup
        with torch.no_grad():
            _ = model(**inputs)

        # Measure throughput
        batch_size = self.scenario.batch_size
        num_iterations = 10

        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(**inputs)

                if self.device == "cuda":
                    torch.cuda.synchronize()

        total_time = time.time() - start_time
        avg_time = total_time / num_iterations

        # Calculate throughput (tokens per second)
        num_tokens = inputs["input_ids"].shape[1]
        throughput = (num_tokens * num_iterations) / total_time

        # Get memory usage
        if self.device == "cuda":
            memory_bytes = torch.cuda.max_memory_allocated()
            memory_gb = memory_bytes / (1024**3)
        else:
            # Estimate for CPU
            memory_gb = param_count * 4 / (1024**3)  # 4 bytes per float32

        latency_ms = avg_time * 1000

        return MeasuredMetrics(
            model_name=model_name,
            parameters_b=parameters_b,
            memory_gb=memory_gb,
            throughput_toks_per_sec=throughput,
            latency_ms_per_request=latency_ms,
            batch_size=batch_size,
        )

    def _apply_pruning(self, model: Any, method: str) -> Any:
        """Apply pruning to a PyTorch model.

        Args:
            model: PyTorch model.
            method: Pruning method.

        Returns:
            Pruned model.
        """

        target_sparsity = self.strategy.parameter_reduction_fraction

        if method == "magnitude":
            # Apply magnitude-based pruning to linear layers
            for _name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(  # type: ignore[no-untyped-call]
                        module, name="weight", amount=target_sparsity
                    )
                    # Make pruning permanent
                    prune.remove(module, "weight")  # type: ignore[no-untyped-call]

        return model

    def _simulate_baseline_measurement(self, model_name: str) -> MeasuredMetrics:
        """Simulate baseline measurement when real model unavailable.

        This generates realistic measurements based on the scenario.
        """
        # Use scenario values with small random variance
        import random

        def variance(x: float, pct: float) -> float:
            return x * (1 + random.uniform(-pct, pct))

        return MeasuredMetrics(
            model_name=f"{model_name} (simulated)",
            parameters_b=self.scenario.parameters_b,
            memory_gb=variance(self.scenario.memory_gb, 0.05),
            throughput_toks_per_sec=variance(
                self.scenario.throughput_toks_per_sec, 0.10
            ),
            latency_ms_per_request=variance(
                self.scenario.tokens_per_request
                / self.scenario.throughput_toks_per_sec
                * 1000,
                0.10,
            ),
            batch_size=self.scenario.batch_size,
            power_watts=self.scenario.power_watts,
        )

    def _simulate_optimized_measurement(self, model_name: str) -> MeasuredMetrics:
        """Simulate optimized measurement when real model unavailable."""

        def variance(x: float, pct: float) -> float:
            return x * (1 + random.uniform(-pct, pct))

        # Apply optimization effects
        mem_reduction = self.strategy.memory_reduction_fraction
        throughput_improvement = self.strategy.throughput_improvement_fraction

        # Parameter reduction affects memory
        optimized_memory = self.scenario.memory_gb * (1 - mem_reduction)
        # Throughput improvement
        optimized_throughput = self.scenario.throughput_toks_per_sec * (
            1 + throughput_improvement
        )

        # Parameter count after pruning
        param_reduction = self.strategy.parameter_reduction_fraction
        optimized_params = self.scenario.parameters_b * (1 - param_reduction)

        return MeasuredMetrics(
            model_name=f"{model_name} (pruned, simulated)",
            parameters_b=variance(optimized_params, 0.05),
            memory_gb=variance(optimized_memory, 0.08),
            throughput_toks_per_sec=variance(optimized_throughput, 0.12),
            latency_ms_per_request=variance(
                self.scenario.tokens_per_request / optimized_throughput * 1000, 0.12
            ),
            batch_size=self.scenario.batch_size,
            power_watts=self.scenario.power_watts
            * (1 - self.strategy.power_reduction_fraction),
        )

    def run_validation(
        self,
        model_name: str | None = None,
        pruning_method: str = "magnitude",
    ) -> ValidationResult:
        """Run full validation comparing Atropos vs measured metrics.

        Args:
            model_name: Model to validate (optional).
            pruning_method: Pruning method to apply.

        Returns:
            ValidationResult with comparison analysis.
        """
        # Measure baseline
        print("Measuring baseline model...")
        baseline = self.measure_baseline(model_name)

        # Measure optimized
        print("Measuring optimized model...")
        optimized = self.measure_optimized(model_name, pruning_method)

        # Build comparisons
        comparisons = self._build_comparisons(optimized)

        # Calculate savings
        atropos_savings = (
            self._atropos_outcome.baseline_annual_total_cost_usd
            - self._atropos_outcome.optimized_annual_total_cost_usd
        ) / self._atropos_outcome.baseline_annual_total_cost_usd * 100

        measured_cost_baseline = baseline.memory_gb  # Proxy for cost
        measured_cost_optimized = optimized.memory_gb
        measured_savings = (
            (measured_cost_baseline - measured_cost_optimized)
            / measured_cost_baseline
            * 100
        )

        # Generate assessment
        assessment = self._generate_assessment(comparisons)

        return ValidationResult(
            scenario_name=self.scenario.name,
            strategy_name=self.strategy.name,
            baseline_metrics=baseline,
            optimized_metrics=optimized,
            comparisons=comparisons,
            atropos_savings_pct=atropos_savings,
            measured_savings_pct=measured_savings,
            overall_assessment=assessment,
        )

    def _build_comparisons(
        self, optimized: MeasuredMetrics
    ) -> list[ComparisonMetric]:
        """Build comparison metrics between Atropos and measured."""
        comparisons = []

        # Memory comparison
        projected_memory = self._atropos_outcome.optimized_memory_gb
        measured_memory = optimized.memory_gb
        comparisons.append(
            ComparisonMetric(
                name="Memory",
                projected=projected_memory,
                measured=measured_memory,
                unit="GB",
                variance_pct=_compute_variance(projected_memory, measured_memory),
            )
        )

        # Throughput comparison
        projected_throughput = (
            self._atropos_outcome.optimized_throughput_toks_per_sec
        )
        measured_throughput = optimized.throughput_toks_per_sec
        comparisons.append(
            ComparisonMetric(
                name="Throughput",
                projected=projected_throughput,
                measured=measured_throughput,
                unit="tok/s",
                variance_pct=_compute_variance(
                    projected_throughput, measured_throughput
                ),
            )
        )

        # Latency comparison (derived from throughput)
        projected_latency = (
            self.scenario.tokens_per_request
            / self._atropos_outcome.optimized_throughput_toks_per_sec
            * 1000
        )
        measured_latency = optimized.latency_ms_per_request
        comparisons.append(
            ComparisonMetric(
                name="Latency",
                projected=projected_latency,
                measured=measured_latency,
                unit="ms",
                variance_pct=_compute_variance(projected_latency, measured_latency),
            )
        )

        return comparisons

    def _generate_assessment(self, comparisons: list[ComparisonMetric]) -> str:
        """Generate overall assessment based on comparisons."""
        accurate_count = sum(1 for c in comparisons if c.is_accurate)
        total_count = len(comparisons)

        if accurate_count == total_count:
            return (
                f"✅ All {total_count} metrics within tolerance. "
                "Atropos projections are accurate for this scenario."
            )
        elif accurate_count >= total_count // 2:
            return (
                f"⚠️ {accurate_count}/{total_count} metrics within tolerance. "
                "Atropos projections are reasonably accurate but may need calibration."
            )
        else:
            return (
                f"❌ Only {accurate_count}/{total_count} metrics within tolerance. "
                "Significant variance detected - scenario may need recalibration."
            )


class ImportErrorException(Exception):  # noqa: N818
    """Exception raised when required packages not installed."""


def run_validation(
    scenario: DeploymentScenario,
    strategy: OptimizationStrategy,
    model_name: str | None = None,
    device: str = "cpu",
    pruning_method: str = "magnitude",
) -> ValidationResult:
    """Convenience function to run validation.

    Args:
        scenario: Deployment scenario.
        strategy: Optimization strategy.
        model_name: Model to validate (optional).
        device: Device to run on.
        pruning_method: Pruning method.

    Returns:
        ValidationResult.
    """
    validator = ModelValidator(scenario, strategy, device)
    return validator.run_validation(model_name, pruning_method)
