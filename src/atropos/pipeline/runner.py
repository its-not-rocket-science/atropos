"""Pipeline runner implementation with stage execution logic."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import DeploymentScenario, OptimizationStrategy

from ..calculations import estimate_outcome
from ..models import QualityRisk
from .config import PipelineConfig
from .models import PipelineResult, PipelineStage, StageResult, StageStatus


def _now() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _risk_rank(risk: QualityRisk) -> int:
    """Convert risk level to numeric rank for comparison."""
    return {"low": 1, "medium": 2, "high": 3}.get(risk, 3)


class PipelineRunner:
    """Runner for executing Atropos pipeline stages."""

    def __init__(self, config: PipelineConfig, dry_run: bool = False):
        """Initialize pipeline runner.

        Args:
            config: Pipeline configuration.
            dry_run: If True, stages are simulated without actual execution.
        """
        self.config = config
        self.dry_run = dry_run
        self._result = PipelineResult(
            pipeline_name=config.name,
            scenario_name="",
            strategy_name="",
        )

    def run(
        self,
        scenario: DeploymentScenario,
        strategy: OptimizationStrategy,
        grid_co2e: float = 0.35,
    ) -> PipelineResult:
        """Execute the full pipeline.

        Args:
            scenario: Deployment scenario to optimize.
            strategy: Optimization strategy to apply.
            grid_co2e: Grid carbon intensity factor.

        Returns:
            PipelineResult with execution details.
        """
        self._result.scenario_name = scenario.name
        self._result.strategy_name = strategy.name
        self._result.start_time = _now()

        try:
            # Stage 1: Assess
            assess_result = self._run_assess(scenario, strategy, grid_co2e)
            self._result.stages.append(assess_result)

            if assess_result.status != StageStatus.SUCCESS:
                self._finalize(StageStatus.FAILED, "Assessment failed")
                return self._result

            # Stage 2: Gate
            gate_result = self._run_gate()
            self._result.stages.append(gate_result)

            if gate_result.status == StageStatus.SKIPPED:
                msg = "Pipeline passed but gated - thresholds not met"
                self._finalize(StageStatus.SUCCESS, msg)
                return self._result

            if gate_result.status != StageStatus.SUCCESS:
                self._finalize(StageStatus.FAILED, "Gate check failed")
                return self._result

            # Stage 3: Prune
            prune_result = self._run_prune(scenario)
            self._result.stages.append(prune_result)

            if prune_result.status != StageStatus.SUCCESS:
                self._finalize(StageStatus.FAILED, "Pruning failed")
                return self._result

            # Stage 4: Recover (if enabled)
            if self.config.recovery and self.config.recovery.enabled:
                recover_result = self._run_recover()
                self._result.stages.append(recover_result)

                if recover_result.status != StageStatus.SUCCESS:
                    self._finalize(StageStatus.FAILED, "Recovery/fine-tuning failed")
                    return self._result

            # Stage 5: Validate
            validate_result = self._run_validate(scenario)
            self._result.stages.append(validate_result)

            if validate_result.status != StageStatus.SUCCESS:
                # Trigger rollback on validation failure
                rollback_result = self._run_rollback()
                self._result.stages.append(rollback_result)
                self._finalize(StageStatus.FAILED, "Validation failed - rollback executed")
                return self._result

            # Stage 6: Deploy (if auto_deploy enabled)
            if self.config.deployment and self.config.deployment.auto_deploy:
                deploy_result = self._run_deploy()
                self._result.stages.append(deploy_result)

                if deploy_result.status != StageStatus.SUCCESS:
                    # Trigger rollback on deployment failure
                    rollback_result = self._run_rollback()
                    self._result.stages.append(rollback_result)
                    self._finalize(StageStatus.FAILED, "Deployment failed - rollback executed")
                    return self._result

            self._finalize(StageStatus.SUCCESS, "Pipeline completed successfully")
            return self._result

        except Exception as e:
            self._finalize(StageStatus.FAILED, f"Pipeline failed with error: {e}")
            return self._result

    def _run_assess(
        self,
        scenario: DeploymentScenario,
        strategy: OptimizationStrategy,
        grid_co2e: float,
    ) -> StageResult:
        """Execute Assess stage: Run ROI analysis."""
        result = StageResult(
            stage=PipelineStage.ASSESS,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        try:
            outcome = estimate_outcome(scenario, strategy, grid_co2e_kg_per_kwh=grid_co2e)
            self._result.roi_outcome = outcome

            result.status = StageStatus.SUCCESS
            if outcome.break_even_years:
                savings = outcome.annual_total_savings_usd
                be_years = outcome.break_even_years
                result.message = (
                    f"ROI complete: ${savings:,.0f}/year savings, {be_years:.1f} year break-even"
                )
            else:
                result.message = "ROI complete: no break-even projected"
            result.metrics = {
                "annual_savings_usd": outcome.annual_total_savings_usd,
                "break_even_years": outcome.break_even_years,
                "quality_risk": outcome.quality_risk,
                "baseline_cost": outcome.baseline_annual_total_cost_usd,
                "optimized_cost": outcome.optimized_annual_total_cost_usd,
            }
        except Exception as e:
            result.status = StageStatus.FAILED
            result.message = f"Assessment failed: {e}"

        result.end_time = _now()
        return result

    def _run_gate(self) -> StageResult:
        """Execute Gate stage: Check thresholds."""
        assert self.config.thresholds is not None, "Thresholds config required"

        result = StageResult(
            stage=PipelineStage.GATE,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        if not self._result.roi_outcome:
            result.status = StageStatus.FAILED
            result.message = "Gate check failed: No ROI outcome from assessment"
            result.end_time = _now()
            return result

        outcome = self._result.roi_outcome
        thresholds = self.config.thresholds

        failures: list[str] = []

        # Check break-even
        if outcome.break_even_years is None:
            failures.append("No break-even projected")
        elif outcome.break_even_years > thresholds.max_break_even_months / 12:
            failures.append(
                f"Break-even time ({outcome.break_even_years:.1f} years) exceeds threshold "
                f"({thresholds.max_break_even_months / 12:.1f} years)"
            )

        # Check annual savings
        if outcome.annual_total_savings_usd < thresholds.min_annual_savings_usd:
            failures.append(
                f"Annual savings ({outcome.annual_total_savings_usd:,.0f} USD) below threshold "
                f"({thresholds.min_annual_savings_usd:,.0f} USD)"
            )

        # Check quality risk
        risk = outcome.quality_risk
        max_risk = thresholds.max_quality_risk
        if _risk_rank(risk) > _risk_rank(max_risk):
            failures.append(f"Quality risk ({risk}) exceeds threshold ({max_risk})")

        if failures:
            result.status = StageStatus.SKIPPED
            result.message = "Gate check failed - pipeline stopped:\n  - " + "\n  - ".join(failures)
            result.metrics = {"failures": failures}
        else:
            result.status = StageStatus.SUCCESS
            result.message = "All thresholds passed - proceeding with optimization"

        result.end_time = _now()
        return result

    def _run_prune(self, scenario: DeploymentScenario) -> StageResult:
        """Execute Prune stage: Run pruning framework."""
        assert self.config.pruning is not None, "Pruning config required"

        result = StageResult(
            stage=PipelineStage.PRUNE,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.pruning

        if self.dry_run:
            result.status = StageStatus.SUCCESS
            fw = config.framework
            sparsity = config.target_sparsity
            result.message = f"[DRY RUN] Would execute {fw} pruning at {sparsity:.0%} sparsity"
            result.end_time = _now()
            return result

        # In real implementation, this would call the pruning framework
        # For now, we simulate success or run custom command if provided
        if config.custom_command:
            try:
                # Run custom pruning command
                subprocess.run(
                    config.custom_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Pruning completed using custom command"
            except subprocess.CalledProcessError as e:
                result.status = StageStatus.FAILED
                result.message = f"Pruning failed: {e.stderr}"
        else:
            # Simulated pruning
            result.status = StageStatus.SUCCESS
            fw = config.framework
            sparsity = config.target_sparsity
            result.message = f"Pruning simulated ({fw} at {sparsity:.0%} sparsity)"
            result.metrics = {
                "framework": config.framework,
                "target_sparsity": config.target_sparsity,
                "structured": config.structured,
            }

        result.end_time = _now()
        return result

    def _run_recover(self) -> StageResult:
        """Execute Recover stage: Fine-tuning."""
        assert self.config.recovery is not None, "Recovery config required"

        result = StageResult(
            stage=PipelineStage.RECOVER,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.recovery

        if self.dry_run:
            result.status = StageStatus.SUCCESS
            result.message = f"[DRY RUN] Would run recovery fine-tuning for {config.epochs} epochs"
            result.end_time = _now()
            return result

        if config.custom_command:
            try:
                subprocess.run(
                    config.custom_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Recovery fine-tuning completed"
            except subprocess.CalledProcessError as e:
                result.status = StageStatus.FAILED
                result.message = f"Recovery failed: {e.stderr}"
        else:
            # Simulated recovery
            result.status = StageStatus.SUCCESS
            lr = config.learning_rate
            result.message = f"Recovery simulated ({config.epochs} epochs at lr={lr})"
            result.metrics = {
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
            }

        result.end_time = _now()
        return result

    def _run_validate(self, scenario: DeploymentScenario) -> StageResult:
        """Execute Validate stage: Benchmark and verify."""
        assert self.config.validation is not None, "Validation config required"

        result = StageResult(
            stage=PipelineStage.VALIDATE,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.validation

        if self.dry_run:
            result.status = StageStatus.SUCCESS
            result.message = f"[DRY RUN] Would run {config.quality_benchmark} validation"
            result.end_time = _now()
            return result

        if config.benchmark_command:
            try:
                subprocess.run(
                    config.benchmark_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Validation completed"
            except subprocess.CalledProcessError as e:
                result.status = StageStatus.FAILED
                result.message = f"Validation failed: {e.stderr}"
        else:
            # Simulated validation
            result.status = StageStatus.SUCCESS
            result.message = f"Validation simulated ({config.quality_benchmark} benchmark)"
            result.metrics = {
                "benchmark": config.quality_benchmark,
                "tolerance_percent": config.tolerance_percent,
            }

        result.end_time = _now()
        return result

    def _run_deploy(self) -> StageResult:
        """Execute Deploy stage."""
        assert self.config.deployment is not None, "Deployment config required"

        result = StageResult(
            stage=PipelineStage.DEPLOY,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.deployment

        if self.dry_run:
            result.status = StageStatus.SUCCESS
            result.message = f"[DRY RUN] Would deploy with {config.canary_percent}% canary"
            result.end_time = _now()
            return result

        if config.deployment_command:
            try:
                subprocess.run(
                    config.deployment_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Deployment completed"
            except subprocess.CalledProcessError as e:
                result.status = StageStatus.FAILED
                result.message = f"Deployment failed: {e.stderr}"
        else:
            result.status = StageStatus.WARNED
            result.message = "No deployment command configured - skipping"

        result.end_time = _now()
        return result

    def _run_rollback(self) -> StageResult:
        """Execute Rollback stage on failure."""
        assert self.config.deployment is not None, "Deployment config required"

        result = StageResult(
            stage=PipelineStage.ROLLBACK,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.deployment

        if self.dry_run:
            result.status = StageStatus.SUCCESS
            result.message = "[DRY RUN] Would rollback deployment"
            result.end_time = _now()
            return result

        if config.rollback_command:
            try:
                subprocess.run(
                    config.rollback_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Rollback completed"
            except subprocess.CalledProcessError as e:
                result.status = StageStatus.FAILED
                result.message = f"Rollback failed: {e.stderr}"
        else:
            result.status = StageStatus.WARNED
            result.message = "No rollback command configured - manual intervention required"

        result.end_time = _now()
        return result

    def _finalize(self, status: StageStatus, message: str) -> None:
        """Finalize pipeline execution."""
        self._result.final_status = status
        self._result.overall_message = message
        self._result.end_time = _now()


def run_pipeline(
    config: PipelineConfig,
    scenario: DeploymentScenario,
    strategy: OptimizationStrategy,
    grid_co2e: float = 0.35,
    dry_run: bool = False,
) -> PipelineResult:
    """Convenience function to run a pipeline.

    Args:
        config: Pipeline configuration.
        scenario: Deployment scenario.
        strategy: Optimization strategy.
        grid_co2e: Grid carbon intensity factor.
        dry_run: If True, simulate without actual execution.

    Returns:
        PipelineResult with execution details.
    """
    runner = PipelineRunner(config, dry_run=dry_run)
    return runner.run(scenario, strategy, grid_co2e)
