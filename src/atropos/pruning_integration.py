"""Integration with pruning frameworks (LLM-Pruner, Wanda, SparseGPT).

This module provides a unified interface to popular LLM pruning frameworks,
allowing Atropos to execute actual pruning based on calculated ROI projections.
"""

from __future__ import annotations

import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .models import DeploymentScenario, OptimizationStrategy
    from .pipeline.config import PipelineConfig


@dataclass
class PruningResult:
    """Result of a pruning operation."""

    success: bool
    original_params: int | None = None
    pruned_params: int | None = None
    sparsity_achieved: float | None = None
    output_path: Path | None = None
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def parameter_reduction_fraction(self) -> float | None:
        """Calculate parameter reduction fraction."""
        if self.original_params and self.pruned_params:
            return (self.original_params - self.pruned_params) / self.original_params
        return None


class PruningFramework(ABC):
    """Abstract base class for pruning framework integrations."""

    name: str = ""
    description: str = ""

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pruning framework.

        Args:
            config: Pipeline configuration with pruning settings.
        """
        self.config = config
        self._check_availability()

    @abstractmethod
    def _check_availability(self) -> bool:
        """Check if the framework is installed and available.

        Returns:
            True if framework is available.

        Raises:
            RuntimeError: If framework is not installed.
        """
        raise NotImplementedError

    @abstractmethod
    def prune(
        self,
        model_name: str,
        output_path: Path,
        target_sparsity: float,
        **kwargs: Any,
    ) -> PruningResult:
        """Execute pruning on a model.

        Args:
            model_name: Name or path of the model to prune.
            output_path: Where to save the pruned model.
            target_sparsity: Target sparsity level (0-1).
            **kwargs: Framework-specific arguments.

        Returns:
            PruningResult with details of the operation.
        """
        raise NotImplementedError

    @abstractmethod
    def estimate_pruning_time(self, model_params_b: float) -> float:
        """Estimate time required for pruning.

        Args:
            model_params_b: Model size in billions of parameters.

        Returns:
            Estimated time in minutes.
        """
        raise NotImplementedError


class LLMPrunerFramework(PruningFramework):
    """Integration with LLM-Pruner for structured pruning."""

    name = "llm-pruner"
    description = "Structured pruning with recovery for LLMs"

    def _check_availability(self) -> bool:
        """Check if LLM-Pruner is installed."""
        try:
            import llm_pruner  # noqa: F401

            return True
        except ImportError as err:
            raise RuntimeError(
                "LLM-Pruner not installed. Install with:\n"
                "  pip install llm-pruner\n"
                "Or clone from: https://github.com/horseee/LLM-Pruner"
            ) from err

    def prune(
        self,
        model_name: str,
        output_path: Path,
        target_sparsity: float,
        **kwargs: Any,
    ) -> PruningResult:
        """Prune model using LLM-Pruner."""
        try:
            # Import here to fail gracefully if not available
            import llm_pruner
            import torch

            # Load model
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading model: {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            original_params = sum(p.numel() for p in model.parameters())

            # Apply LLM-Pruner
            print(f"Applying LLM-Pruner with target sparsity: {target_sparsity:.2%}")

            # Configure pruner based on target sparsity
            pruner = llm_pruner.Pruner(
                model=model,
                tokenizer=tokenizer,
                target_sparsity=target_sparsity,
                **kwargs,
            )

            # Execute pruning
            pruned_model = pruner.prune()

            pruned_params = sum(p.numel() for p in pruned_model.parameters())

            # Save pruned model
            output_path.mkdir(parents=True, exist_ok=True)
            pruned_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)

            return PruningResult(
                success=True,
                original_params=original_params,
                pruned_params=pruned_params,
                sparsity_achieved=target_sparsity,
                output_path=output_path,
                metadata={"framework": "llm-pruner"},
            )

        except Exception as e:
            return PruningResult(
                success=False,
                error_message=str(e),
            )

    def estimate_pruning_time(self, model_params_b: float) -> float:
        """Estimate pruning time based on model size."""
        # LLM-Pruner takes roughly:
        # - 1-7B models: 30-60 minutes
        # - 7-13B models: 60-120 minutes
        # - 13B+ models: 120+ minutes
        if model_params_b <= 1:
            return 30
        elif model_params_b <= 7:
            return 60
        elif model_params_b <= 13:
            return 120
        else:
            return 240


class WandaFramework(PruningFramework):
    """Integration with Wanda (Pruning by Weights And Activations)."""

    name = "wanda"
    description = "Pruning based on weights and activations without retraining"

    def _check_availability(self) -> bool:
        """Check if Wanda is available."""
        # Wanda is typically used as a script, not a package
        wanda_path = self._find_wanda_path()
        if wanda_path:
            return True

        raise RuntimeError(
            "Wanda not found. Clone from:\n"
            "  git clone https://github.com/locuslab/wanda.git\n"
            "Then provide path via WANDA_PATH environment variable."
        )

    def _find_wanda_path(self) -> Path | None:
        """Find Wanda installation path."""
        # Check environment variable
        env_path = Path.home() / "wanda"
        if env_path.exists():
            return env_path

        # Check common locations
        common_paths = [
            Path.home() / "wanda",
            Path("/opt/wanda"),
            Path.home() / "projects" / "wanda",
            Path.home() / "workspace" / "wanda",
        ]
        for path in common_paths:
            if path.exists():
                return path

        return None

    def prune(
        self,
        model_name: str,
        output_path: Path,
        target_sparsity: float,
        **kwargs: Any,
    ) -> PruningResult:
        """Prune model using Wanda."""
        wanda_path = self._find_wanda_path()
        if not wanda_path:
            return PruningResult(
                success=False,
                error_message="Wanda not found",
            )

        # Build Wanda command
        cmd = [
            sys.executable,
            str(wanda_path / "main.py"),
            "--model",
            model_name,
            "--prune_method",
            "wanda",
            "--sparsity",
            str(target_sparsity),
            "--save",
            str(output_path),
        ]

        # Add optional args
        if "nsamples" in kwargs:
            cmd.extend(["--nsamples", str(kwargs["nsamples"])])
        if "seed" in kwargs:
            cmd.extend(["--seed", str(kwargs["seed"])])

        try:
            print(f"Running Wanda: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )

            if result.returncode == 0:
                # Parse output for parameter counts
                # Wanda typically outputs this info
                output_lines = result.stdout.split("\n")
                original_params = None
                pruned_params = None

                for line in output_lines:
                    if "original params:" in line.lower():
                        try:
                            original_params = int(line.split(":")[1].strip().replace(",", ""))
                        except (ValueError, IndexError):
                            pass
                    elif "pruned params:" in line.lower():
                        try:
                            pruned_params = int(line.split(":")[1].strip().replace(",", ""))
                        except (ValueError, IndexError):
                            pass

                return PruningResult(
                    success=True,
                    original_params=original_params,
                    pruned_params=pruned_params,
                    sparsity_achieved=target_sparsity,
                    output_path=output_path,
                    metadata={
                        "framework": "wanda",
                        "command": " ".join(cmd),
                        "stdout": result.stdout,
                    },
                )
            else:
                return PruningResult(
                    success=False,
                    error_message=f"Wanda failed with code {result.returncode}: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            return PruningResult(
                success=False,
                error_message="Wanda pruning timed out after 2 hours",
            )
        except Exception as e:
            return PruningResult(
                success=False,
                error_message=str(e),
            )

    def estimate_pruning_time(self, model_params_b: float) -> float:
        """Estimate pruning time based on model size."""
        # Wanda is faster than LLM-Pruner as it doesn't require recovery
        if model_params_b <= 1:
            return 10
        elif model_params_b <= 7:
            return 30
        elif model_params_b <= 13:
            return 60
        else:
            return 120


class SparseGPTFramework(PruningFramework):
    """Integration with SparseGPT for one-shot pruning."""

    name = "sparsegpt"
    description = "Accurate one-shot pruning for GPT-style models"

    def _check_availability(self) -> bool:
        """Check if SparseGPT is available."""
        try:
            import sparsegpt  # noqa: F401

            return True
        except ImportError as err:
            # Also check for standalone implementation
            sparsegpt_path = Path.home() / "SparseGPT"
            if sparsegpt_path.exists():
                return True

            raise RuntimeError(
                "SparseGPT not found. Options:\n"
                "1. pip install sparsegpt\n"
                "2. Clone from: https://github.com/IST-DASLab/sparsegpt"
            ) from err

    def prune(
        self,
        model_name: str,
        output_path: Path,
        target_sparsity: float,
        **kwargs: Any,
    ) -> PruningResult:
        """Prune model using SparseGPT."""
        try:
            # Try importing sparsegpt package first
            try:
                import sparsegpt
                import torch
                from transformers import AutoModelForCausalLM

                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

                original_params = sum(p.numel() for p in model.parameters())

                # Apply SparseGPT
                pruner = sparsegpt.SparseGPT(
                    model,
                    sparsity=target_sparsity,
                )
                pruner.prune()

                pruned_params = sum(p.numel() for p in model.parameters())

                # Save
                output_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(output_path)

                return PruningResult(
                    success=True,
                    original_params=original_params,
                    pruned_params=pruned_params,
                    sparsity_achieved=target_sparsity,
                    output_path=output_path,
                    metadata={"framework": "sparsegpt"},
                )

            except ImportError:
                # Fall back to subprocess
                return self._prune_via_subprocess(
                    model_name, output_path, target_sparsity, **kwargs
                )

        except Exception as e:
            return PruningResult(
                success=False,
                error_message=str(e),
            )

    def _prune_via_subprocess(
        self,
        model_name: str,
        output_path: Path,
        target_sparsity: float,
        **kwargs: Any,
    ) -> PruningResult:
        """Run SparseGPT via subprocess."""
        sparsegpt_path = Path.home() / "SparseGPT"

        cmd = [
            sys.executable,
            str(sparsegpt_path / "sparsegpt.py"),
            "--model",
            model_name,
            "--sparsity",
            str(target_sparsity),
            "--save",
            str(output_path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )

            if result.returncode == 0:
                return PruningResult(
                    success=True,
                    output_path=output_path,
                    metadata={"framework": "sparsegpt", "method": "subprocess"},
                )
            else:
                return PruningResult(
                    success=False,
                    error_message=result.stderr,
                )

        except Exception as e:
            return PruningResult(
                success=False,
                error_message=str(e),
            )

    def estimate_pruning_time(self, model_params_b: float) -> float:
        """Estimate pruning time based on model size."""
        # SparseGPT is one-shot and relatively fast
        if model_params_b <= 1:
            return 5
        elif model_params_b <= 7:
            return 15
        elif model_params_b <= 13:
            return 30
        else:
            return 60


# Registry of available frameworks
PRUNING_FRAMEWORKS: dict[str, type[PruningFramework]] = {
    "llm-pruner": LLMPrunerFramework,
    "wanda": WandaFramework,
    "sparsegpt": SparseGPTFramework,
}


def get_pruning_framework(
    name: str,
    config: PipelineConfig | None = None,
) -> PruningFramework:
    """Get a pruning framework by name.

    Args:
        name: Framework name (llm-pruner, wanda, sparsegpt).
        config: Optional pipeline configuration.

    Returns:
        PruningFramework instance.

    Raises:
        ValueError: If framework name is unknown.
    """
    if name not in PRUNING_FRAMEWORKS:
        available = ", ".join(PRUNING_FRAMEWORKS.keys())
        raise ValueError(f"Unknown pruning framework '{name}'. Available: {available}")

    framework_class = PRUNING_FRAMEWORKS[name]
    return framework_class(config)


def auto_select_framework(
    scenario: DeploymentScenario,
    strategy: OptimizationStrategy,
) -> str:
    """Automatically select best pruning framework for scenario.

    Args:
        scenario: Deployment scenario.
        strategy: Optimization strategy.

    Returns:
        Recommended framework name.
    """
    params_b = scenario.parameters_b
    reduction = strategy.parameter_reduction_fraction

    # For small models (<1B) with aggressive pruning, use SparseGPT
    if params_b <= 1 and reduction >= 0.4:
        return "sparsegpt"

    # For medium models (1-7B) with structured pruning, use LLM-Pruner
    if params_b <= 7 and strategy.name in ("structured_pruning", "hardware_aware_pruning"):
        return "llm-pruner"

    # For quick experiments, use Wanda (no recovery needed)
    if reduction <= 0.3:
        return "wanda"

    # Default to LLM-Pruner for best quality
    return "llm-pruner"


def run_pruning_pipeline(
    model_name: str,
    output_dir: Path,
    target_sparsity: float,
    framework: str | None = None,
    config: PipelineConfig | None = None,
    **kwargs: Any,
) -> PruningResult:
    """Execute pruning with automatic framework selection.

    Args:
        model_name: Model to prune.
        output_dir: Output directory for pruned model.
        target_sparsity: Target sparsity level.
        framework: Framework to use (auto-selected if None).
        config: Pipeline configuration.
        **kwargs: Additional framework arguments.

    Returns:
        PruningResult with operation details.
    """
    if framework is None and config and config.pruning:
        framework = config.pruning.framework

    if framework is None:
        raise ValueError("No pruning framework specified")

    framework_obj = get_pruning_framework(framework, config)

    print(f"Pruning {model_name} with {framework} (target sparsity: {target_sparsity:.2%})")

    return framework_obj.prune(
        model_name=model_name,
        output_path=output_dir,
        target_sparsity=target_sparsity,
        **kwargs,
    )
