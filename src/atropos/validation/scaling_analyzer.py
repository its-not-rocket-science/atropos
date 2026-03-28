"""Multi-GPU scaling analysis for Atropos validation.

Provides MultiGPUScalingAnalyzer for analyzing scaling efficiency across
different GPU counts, identifying bottlenecks, and generating optimization
recommendations.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, cast

from ..models import DeploymentScenario, OptimizationStrategy
from .distributed_benchmark import DistributedBenchmarkWrapper


@dataclass
class ScalingAnalysisResult:
    """Results of multi-GPU scaling analysis."""

    scenario_name: str
    strategy_name: str
    model_name: str
    gpu_counts: list[int]
    throughputs: dict[int, float]  # GPU count -> throughput (tok/s)
    memories: dict[int, float]  # GPU count -> memory per GPU (GB)
    scaling_efficiencies: dict[int, float | None]  # GPU count -> scaling efficiency
    ideal_linear_throughputs: dict[int, float]  # GPU count -> ideal linear scaling
    bottlenecks: list[str]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_markdown(self) -> str:
        """Generate markdown report."""
        lines = [
            f"# Multi-GPU Scaling Analysis: {self.scenario_name}",
            f"**Model**: {self.model_name}",
            f"**Strategy**: {self.strategy_name}",
            "",
            "## Scaling Results",
            "",
            "| GPU Count | Throughput (tok/s) | Memory per GPU (GB) | Scaling Efficiency |",
            "|-----------|-------------------|---------------------|-------------------|",
        ]

        for count in self.gpu_counts:
            throughput = self.throughputs.get(count, 0.0)
            memory = self.memories.get(count, 0.0)
            efficiency = self.scaling_efficiencies.get(count)
            efficiency_str = f"{efficiency:.1%}" if efficiency is not None else "N/A"
            lines.append(f"| {count} | {throughput:.1f} | {memory:.2f} | {efficiency_str} |")

        lines.extend(
            [
                "",
                "## Bottlenecks Identified",
                "",
            ]
        )

        if self.bottlenecks:
            for bottleneck in self.bottlenecks:
                lines.append(f"- {bottleneck}")
        else:
            lines.append("No significant bottlenecks detected.")

        lines.extend(
            [
                "",
                "## Recommendations",
                "",
            ]
        )

        for rec in self.recommendations:
            lines.append(f"- {rec}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Generate JSON report."""
        return json.dumps(self.to_dict(), indent=2)


class MultiGPUScalingAnalyzer:
    """Analyze scaling efficiency across GPU counts."""

    def __init__(
        self,
        scenario: DeploymentScenario,
        strategy: OptimizationStrategy,
        model_name: str,
        device: str = "cuda",
        max_gpus: int = 8,
        gpu_counts: list[int] | None = None,
    ):
        """Initialize scaling analyzer.

        Args:
            scenario: Deployment scenario.
            strategy: Optimization strategy.
            model_name: HuggingFace model name.
            device: Device to run on (default: "cuda").
            max_gpus: Maximum GPU count to test (default: 8).
            gpu_counts: Specific GPU counts to test (default: powers of 2 up to max_gpus).
        """
        self.scenario = scenario
        self.strategy = strategy
        self.model_name = model_name
        self.device = device
        self.max_gpus = max_gpus

        if gpu_counts is None:
            # Default to powers of 2: 1, 2, 4, 8, ...
            self.gpu_counts = [2**i for i in range(0, max_gpus.bit_length()) if 2**i <= max_gpus]
        else:
            self.gpu_counts = sorted(gpu_counts)

        # Ensure 1 GPU is included for baseline
        if 1 not in self.gpu_counts:
            self.gpu_counts.insert(0, 1)

    def run_analysis(self) -> ScalingAnalysisResult:
        """Run scaling analysis across GPU counts.

        Returns:
            ScalingAnalysisResult with metrics, bottlenecks, and recommendations.
        """
        # Import deferred to avoid Windows CUDA deadlock
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model = model.to(self.device)  # type: ignore[arg-type]
        model.eval()

        # Initialize benchmark wrapper
        wrapper = DistributedBenchmarkWrapper(
            gpu_count=1,  # Will be overridden per test
            parallel_strategy=self.scenario.parallel_strategy,
            batch_size_per_gpu=self.scenario.batch_size,
            measure_scaling_efficiency=True,
        )

        # Run benchmarks across GPU counts
        scaling_curve = wrapper.compute_scaling_curve(
            model=model,
            tokenizer=tokenizer,
            gpu_counts=self.gpu_counts,
            device=self.device,
            model_name=self.model_name,
            batch_size=self.scenario.batch_size,
            tokens_per_request=self.scenario.tokens_per_request,
        )

        # Extract metrics
        throughputs = {}
        memories = {}
        scaling_efficiencies = {}
        ideal_linear_throughputs = {}

        single_gpu_throughput = cast(
            float, scaling_curve.get(1, {}).get("throughput_toks_per_sec", 0.0)
        )

        for count, metrics in scaling_curve.items():
            throughputs[count] = cast(float, metrics["throughput_toks_per_sec"])
            memories[count] = cast(float, metrics["memory_gb"])
            scaling_efficiencies[count] = metrics.get("scaling_efficiency")
            ideal_linear_throughputs[count] = single_gpu_throughput * count

        # Analyze bottlenecks
        bottlenecks = self._analyze_bottlenecks(scaling_curve)

        # Generate recommendations
        recommendations = self._generate_recommendations(bottlenecks, scaling_curve)

        return ScalingAnalysisResult(
            scenario_name=self.scenario.name,
            strategy_name=self.strategy.name,
            model_name=self.model_name,
            gpu_counts=self.gpu_counts,
            throughputs=throughputs,
            memories=memories,
            scaling_efficiencies=scaling_efficiencies,
            ideal_linear_throughputs=ideal_linear_throughputs,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
        )

    def _analyze_bottlenecks(self, scaling_curve: dict[int, dict[str, float | None]]) -> list[str]:
        """Analyze scaling curve to identify bottlenecks."""
        bottlenecks = []

        # Check scaling efficiency thresholds
        for count, metrics in scaling_curve.items():
            if count <= 1:
                continue

            efficiency = metrics.get("scaling_efficiency")
            if efficiency is not None:
                if efficiency < 0.7:
                    bottlenecks.append(
                        f"Poor scaling efficiency ({efficiency:.1%}) at {count} GPUs: "
                        f"communication overhead may be limiting performance."
                    )
                elif efficiency < 0.85:
                    bottlenecks.append(
                        f"Moderate scaling efficiency ({efficiency:.1%}) at {count} GPUs: "
                        f"consider optimizing data parallelism configuration."
                    )

        # Check memory imbalance
        memory_values = [
            cast(float, metrics.get("memory_gb", 0.0)) for metrics in scaling_curve.values()
        ]
        if len(memory_values) > 1:
            max_memory = max(memory_values)
            min_memory = min(memory_values)
            if max_memory > 0 and (max_memory - min_memory) / max_memory > 0.2:
                bottlenecks.append(
                    "Significant memory imbalance across GPU counts: "
                    "consider uniform batch sizing or model partitioning."
                )

        return bottlenecks

    def _generate_recommendations(
        self,
        bottlenecks: list[str],
        scaling_curve: dict[int, dict[str, float | None]],
    ) -> list[str]:
        """Generate optimization recommendations based on bottlenecks."""
        recommendations = []

        # Check if scaling efficiency decreases with GPU count
        efficiencies = []
        for count, metrics in scaling_curve.items():
            if count > 1:
                eff = metrics.get("scaling_efficiency")
                if eff is not None:
                    efficiencies.append((count, eff))

        if len(efficiencies) >= 2:
            # Sort by GPU count
            efficiencies.sort(key=lambda x: x[0])
            # Check if efficiency drops significantly
            prev_eff = efficiencies[0][1]
            for count, eff in efficiencies[1:]:
                if eff < prev_eff * 0.8:  # More than 20% drop
                    recommendations.append(
                        f"Scaling efficiency drops at {count} GPUs: "
                        "consider using tensor parallelism or model parallelism "
                        "for larger GPU counts."
                    )
                prev_eff = eff

        # Default recommendations
        if not recommendations:
            recommendations.append(
                "Scaling efficiency is good. Consider increasing batch size "
                "for better GPU utilization."
            )

        # Add bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            if "communication overhead" in bottleneck.lower():
                recommendations.append(
                    "Reduce communication overhead by using gradient compression "
                    "or increasing batch size per GPU."
                )
            elif "memory imbalance" in bottleneck.lower():
                recommendations.append(
                    "Balance memory usage by using uniform batch sizes across GPUs "
                    "or implementing dynamic load balancing."
                )

        return recommendations


def analyze_scaling(
    scenario: DeploymentScenario,
    strategy: OptimizationStrategy,
    model_name: str,
    device: str = "cuda",
    max_gpus: int = 8,
    gpu_counts: list[int] | None = None,
) -> ScalingAnalysisResult:
    """Convenience function for scaling analysis."""
    analyzer = MultiGPUScalingAnalyzer(
        scenario=scenario,
        strategy=strategy,
        model_name=model_name,
        device=device,
        max_gpus=max_gpus,
        gpu_counts=gpu_counts,
    )
    return analyzer.run_analysis()
