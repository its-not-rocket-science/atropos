"""Pipeline runner implementation with stage execution logic."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..calculations import estimate_outcome
from ..deployment import get_platform, get_strategy
from ..deployment.models import (
    DeploymentRequest,
    DeploymentStatus,
    DeploymentStrategyType,
)
from ..logging_config import SHOW_TRACEBACK, get_logger
from ..models import QualityRisk
from .config import PipelineConfig
from .models import PipelineResult, PipelineStage, StageResult, StageStatus

logger = get_logger("pipeline")

if TYPE_CHECKING:
    from ..models import DeploymentScenario, OptimizationStrategy


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
        self._deployment_id: str | None = None
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

        logger.info(
            "Starting pipeline execution: scenario=%s, strategy=%s, dry_run=%s",
            scenario.name,
            strategy.name,
            self.dry_run,
        )

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
            logger.error("Pipeline failed with unexpected error: %s", e, exc_info=SHOW_TRACEBACK)
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
            logger.debug("Running ROI assessment")
            outcome = estimate_outcome(scenario, strategy, grid_co2e_kg_per_kwh=grid_co2e)
            self._result.roi_outcome = outcome

            result.status = StageStatus.SUCCESS
            if outcome.break_even_years:
                savings = outcome.annual_total_savings_usd
                be_years = outcome.break_even_years
                result.message = (
                    f"ROI complete: ${savings:,.0f}/year savings, {be_years:.1f} year break-even"
                )
                logger.info(
                    "Assessment completed: $%s/year savings, %s year break-even",
                    savings,
                    be_years,
                )
            else:
                result.message = "ROI complete: no break-even projected"
                logger.info("Assessment completed: no break-even projected")
            result.metrics = {
                "annual_savings_usd": outcome.annual_total_savings_usd,
                "break_even_years": outcome.break_even_years,
                "quality_risk": outcome.quality_risk,
                "baseline_cost": outcome.baseline_annual_total_cost_usd,
                "optimized_cost": outcome.optimized_annual_total_cost_usd,
            }
        except Exception as e:
            logger.error("Assessment failed: %s", e, exc_info=SHOW_TRACEBACK)
            result.status = StageStatus.FAILED
            result.message = f"Assessment failed: {e}"

        result.end_time = _now()
        return result

    def _run_gate(self) -> StageResult:
        """Execute Gate stage: Check thresholds."""
        assert self.config.thresholds is not None, "Thresholds config required"
        logger.debug("Running Gate stage")

        result = StageResult(
            stage=PipelineStage.GATE,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        if not self._result.roi_outcome:
            logger.error("Gate check failed: No ROI outcome from assessment")
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
            logger.info("Gate check failed: %s", ", ".join(failures))
        else:
            result.status = StageStatus.SUCCESS
            result.message = "All thresholds passed - proceeding with optimization"
            logger.info("Gate check passed - proceeding with optimization")

        result.end_time = _now()
        return result

    def _run_prune(self, scenario: DeploymentScenario) -> StageResult:
        """Execute Prune stage: Run pruning framework."""
        assert self.config.pruning is not None, "Pruning config required"
        logger.debug("Running Prune stage")

        result = StageResult(
            stage=PipelineStage.PRUNE,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.pruning

        if self.dry_run:
            logger.debug("Dry run: would execute pruning")
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
                logger.debug("Running custom pruning command: %s", config.custom_command)
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
                logger.info("Pruning completed using custom command")
            except subprocess.CalledProcessError as e:
                logger.error("Pruning failed: %s", e.stderr, exc_info=SHOW_TRACEBACK)
                result.status = StageStatus.FAILED
                result.message = f"Pruning failed: {e.stderr}"
        else:
            # Simulated pruning
            result.status = StageStatus.SUCCESS
            fw = config.framework
            sparsity = config.target_sparsity
            result.message = f"Pruning simulated ({fw} at {sparsity:.0%} sparsity)"
            logger.info("Pruning simulated successfully: %s at %s sparsity", fw, f"{sparsity:.0%}")
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
        logger.debug("Running Recover stage")

        result = StageResult(
            stage=PipelineStage.RECOVER,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.recovery

        if self.dry_run:
            logger.debug("Dry run: would run recovery fine-tuning")
            result.status = StageStatus.SUCCESS
            result.message = f"[DRY RUN] Would run recovery fine-tuning for {config.epochs} epochs"
            result.end_time = _now()
            return result

        if config.custom_command:
            try:
                logger.debug("Running custom recovery command: %s", config.custom_command)
                subprocess.run(
                    config.custom_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Recovery fine-tuning completed"
                logger.info("Recovery fine-tuning completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error("Recovery failed: %s", e.stderr, exc_info=SHOW_TRACEBACK)
                result.status = StageStatus.FAILED
                result.message = f"Recovery failed: {e.stderr}"
        else:
            # Simulated recovery
            result.status = StageStatus.SUCCESS
            lr = config.learning_rate
            result.message = f"Recovery simulated ({config.epochs} epochs at lr={lr})"
            logger.info("Recovery simulated: %s epochs at lr=%s", config.epochs, lr)
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
        logger.debug("Running Validate stage")

        result = StageResult(
            stage=PipelineStage.VALIDATE,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.validation

        if self.dry_run:
            logger.debug("Dry run: would run validation")
            result.status = StageStatus.SUCCESS
            result.message = f"[DRY RUN] Would run {config.quality_benchmark} validation"
            result.end_time = _now()
            return result

        if config.benchmark_command:
            try:
                logger.debug("Running benchmark command: %s", config.benchmark_command)
                subprocess.run(
                    config.benchmark_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Validation completed"
                logger.info("Validation completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error("Validation failed: %s", e.stderr, exc_info=SHOW_TRACEBACK)
                result.status = StageStatus.FAILED
                result.message = f"Validation failed: {e.stderr}"
        else:
            # Simulated validation
            result.status = StageStatus.SUCCESS
            result.message = f"Validation simulated ({config.quality_benchmark} benchmark)"
            logger.info("Validation simulated: %s benchmark", config.quality_benchmark)
            result.metrics = {
                "benchmark": config.quality_benchmark,
                "tolerance_percent": config.tolerance_percent,
            }

        result.end_time = _now()
        return result

    def _run_deploy(self) -> StageResult:
        """Execute Deploy stage."""
        assert self.config.deployment is not None, "Deployment config required"
        logger.debug("Running Deploy stage")

        result = StageResult(
            stage=PipelineStage.DEPLOY,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.deployment

        if self.dry_run:
            logger.debug("Dry run: would deploy")
            result.status = StageStatus.SUCCESS
            if config.platform:
                result.message = (
                    f"[DRY RUN] Would deploy using {config.platform} platform "
                    f"with {config.strategy} strategy"
                )
            elif config.deployment_command:
                result.message = f"[DRY RUN] Would deploy with command: {config.deployment_command}"
            else:
                result.message = (
                    "[DRY RUN] Would skip deployment (no platform or command configured)"
                )
            result.end_time = _now()
            return result

        # Determine model path
        model_path = config.model_path
        if not model_path and self.config.pruning and self.config.pruning.output_path:
            model_path = self.config.pruning.output_path
        if not model_path:
            result.status = StageStatus.FAILED
            result.message = "No model path specified and pruning output_path not set"
            result.end_time = _now()
            return result

        # Deployment automation path
        if config.platform:
            try:
                logger.debug(
                    "Deploying via platform %s with strategy %s",
                    config.platform,
                    config.strategy,
                )
                # Create deployment request
                strategy_enum = DeploymentStrategyType[config.strategy.upper().replace("-", "_")]
                request = DeploymentRequest(
                    model_path=model_path,
                    platform=config.platform,
                    strategy=strategy_enum,
                    strategy_config=config.strategy_config,
                    health_checks=config.health_checks,
                    metadata=config.metadata,
                )
                # Get platform and strategy
                platform = get_platform(config.platform, config.platform_config)
                strategy = get_strategy(config.strategy, config.strategy_config)
                # Execute deployment
                deployment_result = strategy.execute(platform, request)
                # Map deployment result to stage result
                if deployment_result.status == DeploymentStatus.SUCCESS:
                    result.status = StageStatus.SUCCESS
                    result.message = deployment_result.message
                    self._deployment_id = deployment_result.deployment_id
                    result.metrics = {
                        "deployment_id": deployment_result.deployment_id,
                        "endpoints": deployment_result.endpoints,
                        "duration_seconds": deployment_result.duration_seconds,
                        **deployment_result.metrics,
                    }
                    logger.info("Deployment completed successfully: %s", deployment_result.message)
                else:
                    result.status = StageStatus.FAILED
                    result.message = deployment_result.message
                    logger.error("Deployment failed: %s", deployment_result.message)
            except Exception as e:
                logger.error("Deployment failed with error: %s", e, exc_info=SHOW_TRACEBACK)
                result.status = StageStatus.FAILED
                result.message = f"Deployment failed: {e}"
        # Legacy shell command path
        elif config.deployment_command:
            try:
                logger.debug("Running deployment command: %s", config.deployment_command)
                subprocess.run(
                    config.deployment_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Deployment completed"
                logger.info("Deployment completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error("Deployment failed: %s", e.stderr, exc_info=SHOW_TRACEBACK)
                result.status = StageStatus.FAILED
                result.message = f"Deployment failed: {e.stderr}"
        else:
            result.status = StageStatus.WARNED
            result.message = "No deployment command or platform configured - skipping"
            logger.warning("No deployment command or platform configured - skipping deployment")

        result.end_time = _now()
        return result

    def _run_rollback(self) -> StageResult:
        """Execute Rollback stage on failure."""
        assert self.config.deployment is not None, "Deployment config required"
        logger.debug("Running Rollback stage")

        result = StageResult(
            stage=PipelineStage.ROLLBACK,
            status=StageStatus.RUNNING,
            start_time=_now(),
        )

        config = self.config.deployment

        if self.dry_run:
            logger.debug("Dry run: would rollback deployment")
            result.status = StageStatus.SUCCESS
            if config.platform and self._deployment_id:
                result.message = (
                    f"[DRY RUN] Would rollback deployment {self._deployment_id} "
                    f"via {config.platform}"
                )
            elif config.rollback_command:
                result.message = f"[DRY RUN] Would rollback with command: {config.rollback_command}"
            else:
                result.message = (
                    "[DRY RUN] Would skip rollback (no platform/deployment_id or command)"
                )
            result.end_time = _now()
            return result

        # Deployment automation rollback path
        if config.platform and self._deployment_id:
            try:
                logger.debug(
                    "Rolling back deployment %s via platform %s",
                    self._deployment_id,
                    config.platform,
                )
                platform = get_platform(config.platform, config.platform_config)
                rollback_result = platform.rollback(self._deployment_id)
                if rollback_result.status == DeploymentStatus.SUCCESS:
                    result.status = StageStatus.SUCCESS
                    result.message = rollback_result.message
                    logger.info("Rollback completed successfully: %s", rollback_result.message)
                else:
                    result.status = StageStatus.FAILED
                    result.message = rollback_result.message
                    logger.error("Rollback failed: %s", rollback_result.message)
            except Exception as e:
                logger.error("Rollback failed with error: %s", e, exc_info=SHOW_TRACEBACK)
                result.status = StageStatus.FAILED
                result.message = f"Rollback failed: {e}"
        # Legacy shell command path
        elif config.rollback_command:
            try:
                logger.debug("Running rollback command: %s", config.rollback_command)
                subprocess.run(
                    config.rollback_command,
                    shell=True,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result.status = StageStatus.SUCCESS
                result.message = "Rollback completed"
                logger.info("Rollback completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error("Rollback failed: %s", e.stderr, exc_info=SHOW_TRACEBACK)
                result.status = StageStatus.FAILED
                result.message = f"Rollback failed: {e.stderr}"
        else:
            result.status = StageStatus.WARNED
            result.message = (
                "No rollback command or platform deployment ID - manual intervention required"
            )
            logger.warning(
                "No rollback command or platform deployment ID - manual intervention required"
            )

        result.end_time = _now()
        return result

    def _finalize(self, status: StageStatus, message: str) -> None:
        """Finalize pipeline execution."""
        self._result.final_status = status
        self._result.overall_message = message
        self._result.end_time = _now()

        log_level = logger.info if status == StageStatus.SUCCESS else logger.error
        log_level("Pipeline finalized with status %s: %s", status.value, message)


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
