"""Experiment storage for A/B testing framework.

Provides persistent storage of experiment configurations, deployment IDs,
metrics, and results using JSON files in ~/.atropos/experiments/.
"""

from __future__ import annotations

import json
import typing
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    ABTestConfig,
    ExperimentResult,
    ExperimentStatus,
    StatisticalResult,
    VariantMetrics,
)


class ExperimentStore:
    """Store for A/B test experiments."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize experiment store.

        Args:
            base_dir: Base directory for experiment storage.
                Defaults to ~/.atropos/experiments.
        """
        if base_dir is None:
            base_dir = Path.home() / ".atropos" / "experiments"
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _experiment_path(self, experiment_id: str) -> Path:
        """Get path to experiment JSON file."""
        return self.base_dir / f"{experiment_id}.json"

    def save_experiment(self, experiment_id: str, data: dict[str, Any]) -> None:
        """Save experiment data to JSON file.

        Args:
            experiment_id: Unique experiment identifier.
            data: Experiment data dictionary.
        """
        path = self._experiment_path(experiment_id)
        # Add/update metadata timestamps
        data["updated_at"] = datetime.now().isoformat()
        if "created_at" not in data:
            data["created_at"] = data["updated_at"]
        # Write to file with atomic write pattern
        temp_path = path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        temp_path.replace(path)

    def load_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Load experiment data from JSON file.

        Args:
            experiment_id: Unique experiment identifier.

        Returns:
            Experiment data dictionary, or None if not found.
        """
        path = self._experiment_path(experiment_id)
        if not path.exists():
            return None
        with open(path) as f:
            return typing.cast(dict[str, Any], json.load(f))

    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete experiment data.

        Args:
            experiment_id: Unique experiment identifier.

        Returns:
            True if experiment was deleted, False if not found.
        """
        path = self._experiment_path(experiment_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_experiments(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        """List all experiments, optionally filtered by status.

        Args:
            status_filter: Status to filter by (draft, running, paused,
                stopped, completed). Case-insensitive.

        Returns:
            List of experiment data dictionaries.
        """
        experiments = []
        for path in self.base_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                if status_filter:
                    exp_status = data.get("status", "").lower()
                    if exp_status != status_filter.lower():
                        continue
                data["experiment_id"] = path.stem
                experiments.append(data)
            except (json.JSONDecodeError, KeyError):
                # Skip corrupted files
                continue
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments

    def save_config(self, config: ABTestConfig) -> None:
        """Save experiment configuration.

        Args:
            config: ABTestConfig instance.
        """
        data = {
            "type": "config",
            "config": config.to_dict(),
            "status": ExperimentStatus.DRAFT.value,
            "deployment_ids": {},
            "variant_metrics": {},
            "statistical_results": {},
        }
        self.save_experiment(config.experiment_id, data)

    def save_result(self, result: ExperimentResult) -> None:
        """Save experiment result.

        Args:
            result: ExperimentResult instance.
        """
        data = {
            "type": "result",
            "result": result.to_dict(),
            "status": result.status.value,
            "updated_at": datetime.now().isoformat(),
        }
        self.save_experiment(result.experiment_id, data)

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: ExperimentStatus | None = None,
        deployment_ids: dict[str, str] | None = None,
        variant_metrics: dict[str, VariantMetrics] | None = None,
        statistical_results: dict[str, StatisticalResult] | None = None,
        winner_variant_id: str | None = None,
        confidence: float | None = None,
        end_time: str | None = None,
        start_time: str | None = None,
    ) -> None:
        """Update experiment fields.

        Args:
            experiment_id: Unique experiment identifier.
            status: New experiment status.
            deployment_ids: Deployment IDs per variant.
            variant_metrics: Updated variant metrics.
            statistical_results: Updated statistical results.
            winner_variant_id: Winning variant ID.
            confidence: Confidence in winner.
            end_time: Experiment end timestamp.
            start_time: Experiment start timestamp.
        """
        data = self.load_experiment(experiment_id)
        if data is None:
            raise KeyError(f"Experiment not found: {experiment_id}")

        if status is not None:
            data["status"] = status.value
        if deployment_ids is not None:
            data["deployment_ids"] = deployment_ids
        if variant_metrics is not None:
            # Convert VariantMetrics to dict
            data["variant_metrics"] = {vid: vm.to_dict() for vid, vm in variant_metrics.items()}
        if statistical_results is not None:
            # Convert StatisticalResult to dict
            data["statistical_results"] = {
                metric: sr.to_dict() for metric, sr in statistical_results.items()
            }
        if winner_variant_id is not None:
            data["winner_variant_id"] = winner_variant_id
        if confidence is not None:
            data["confidence"] = confidence
        if end_time is not None:
            data["end_time"] = end_time
        if start_time is not None:
            data["start_time"] = start_time

        self.save_experiment(experiment_id, data)


# Global default store instance
_default_store: ExperimentStore | None = None


def get_default_store() -> ExperimentStore:
    """Get the default experiment store instance."""
    global _default_store
    if _default_store is None:
        _default_store = ExperimentStore()
    return _default_store
