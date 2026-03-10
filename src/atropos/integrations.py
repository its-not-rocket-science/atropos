"""Integration with experiment trackers (Weights & Biases, MLflow)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

from .models import DeploymentScenario


@dataclass(frozen=True)
class RunInfo:
    """Normalized experiment run information.

    Attributes:
        run_id: Unique identifier for the run.
        experiment: Experiment or project name.
        tracker: Source tracker (wandb, mlflow).
        parameters_b: Model size in billions of parameters (optional).
        memory_gb: Peak memory usage in GB.
        throughput_toks_per_sec: Token throughput per second.
        latency_ms_per_request: Average latency per request in milliseconds.
        power_watts: Power consumption in watts (optional).
        requests_per_day: Request volume (optional).
        tokens_per_request: Average tokens per request.
        metrics: Additional raw metrics from the tracker.
        config: Run configuration/parameters.
        tags: Run tags/labels.
        url: URL to view the run in the tracker UI.
    """

    run_id: str
    experiment: str
    tracker: str
    memory_gb: float
    throughput_toks_per_sec: float
    latency_ms_per_request: float
    tokens_per_request: float
    parameters_b: float | None = None
    power_watts: float | None = None
    requests_per_day: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    url: str | None = None


class ExperimentTracker(ABC):
    """Abstract base class for experiment tracker integrations."""

    def __init__(self, api_key: str | None = None, host: str | None = None):
        """Initialize the tracker.

        Args:
            api_key: API key for authentication (optional, uses env if not provided).
            host: Tracker host URL (optional, uses default if not provided).
        """
        self.api_key = api_key
        self.host = host

    @abstractmethod
    def get_run(self, run_id: str, **kwargs: Any) -> RunInfo:
        """Get a specific run by ID.

        Args:
            run_id: The run identifier.
            **kwargs: Additional tracker-specific arguments.

        Returns:
            RunInfo with normalized run data.

        Raises:
            ValueError: If run not found.
            RuntimeError: If tracker API not available.
        """
        raise NotImplementedError

    @abstractmethod
    def list_runs(
        self,
        experiment: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[RunInfo]:
        """List runs matching criteria.

        Args:
            experiment: Experiment/project name filter.
            filters: Additional filters (tracker-specific).
            limit: Maximum number of runs to return.
            **kwargs: Additional tracker-specific arguments.

        Returns:
            List of RunInfo objects.
        """
        raise NotImplementedError

    def _extract_model_size(self, config: dict[str, Any]) -> float | None:
        """Extract model size from config if available."""
        # Try common config keys
        for key in ["model_size", "model_size_b", "parameters_b", "params_b"]:
            if key in config:
                val = config[key]
                if isinstance(val, (int, float)):
                    return float(val)

        # Try to extract from model name
        model_name = config.get("model", config.get("model_name", ""))
        if isinstance(model_name, str):
            import re

            match = re.search(r"-(\d+)(\.\d+)?b", model_name.lower())
            if match:
                return (
                    float(match.group(1))
                    if match.group(2) is None
                    else float(match.group(1) + match.group(2))
                )

        return None


class WandbTracker(ExperimentTracker):
    """Weights & Biases experiment tracker integration."""

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        entity: str | None = None,
    ):
        """Initialize W&B tracker.

        Args:
            api_key: W&B API key (optional).
            host: W&B host URL (optional).
            entity: W&B entity (user/team) name.
        """
        super().__init__(api_key, host)
        self.entity = entity
        self._api = None

    def _get_api(self) -> Any:
        """Get or create W&B API client."""
        if self._api is None:
            try:
                import wandb

                if self.api_key:
                    wandb.login(key=self.api_key, host=self.host)
                self._api = wandb.Api()
            except ImportError as e:
                raise RuntimeError("wandb not installed. Install with: pip install wandb") from e
        return self._api

    def get_run(self, run_id: str, project: str | None = None, **kwargs: Any) -> RunInfo:
        """Get a specific W&B run by ID.

        Args:
            run_id: The run path (format: entity/project/run_id) or just run_id.
            project: Project name if not included in run_id.
            **kwargs: Additional arguments (ignored).

        Returns:
            RunInfo with normalized run data.
        """
        api = self._get_api()

        # Parse run path
        if "/" in run_id:
            path = run_id
        elif project and self.entity:
            path = f"{self.entity}/{project}/{run_id}"
        else:
            raise ValueError(
                "run_id must be full path (entity/project/run_id) or "
                "provide project with entity set"
            )

        run = api.run(path)

        # Extract metrics
        history = run.history()
        metrics = {}
        if not history.empty:
            # Get latest values for each metric
            for col in history.columns:
                if col not in ["_step", "_runtime", "_timestamp"]:
                    col_data = history[col].dropna()
                    val = col_data.iloc[-1] if not col_data.empty else None
                    if val is not None:
                        metrics[col] = val

        # Calculate throughput and latency from summary if available
        summary = dict(run.summary)
        throughput = summary.get("throughput", summary.get("tok_per_sec", 0.0))
        latency = summary.get("latency", summary.get("mean_latency_ms", 0.0))
        memory = summary.get("gpu_memory_gb", summary.get("memory_gb", 0.0))

        # If not in summary, try to get from metrics history
        if throughput == 0.0 and "throughput" in metrics:
            throughput = metrics["throughput"]
        if latency == 0.0 and "latency" in metrics:
            latency = metrics["latency"]
        if memory == 0.0 and "gpu_memory_gb" in metrics:
            memory = metrics["gpu_memory_gb"]

        # Extract tokens per request from config or metrics
        config = dict(run.config)
        tokens = config.get("tokens_per_request") or config.get("max_tokens") or 1000

        return RunInfo(
            run_id=run.id,
            experiment=run.project,
            tracker="wandb",
            parameters_b=self._extract_model_size(config),
            memory_gb=float(memory) if memory else 0.0,
            throughput_toks_per_sec=float(throughput) if throughput else 0.0,
            latency_ms_per_request=float(latency) if latency else 0.0,
            tokens_per_request=float(tokens),
            power_watts=summary.get("power_watts"),
            requests_per_day=config.get("requests_per_day"),
            metrics=metrics,
            config=config,
            tags=list(run.tags),
            url=run.url,
        )

    def list_runs(
        self,
        experiment: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[RunInfo]:
        """List W&B runs matching criteria.

        Args:
            experiment: Project name.
            filters: W&B filters dict (e.g., {"config.model": "llama"}).
            limit: Maximum number of runs.
            **kwargs: Additional arguments including 'entity'.

        Returns:
            List of RunInfo objects.
        """
        api = self._get_api()

        if not experiment:
            raise ValueError("experiment (project name) is required for wandb")

        entity = kwargs.get("entity", self.entity)
        if not entity:
            raise ValueError("entity is required for wandb list_runs")

        # Build filter string
        filter_str = None
        if filters:
            filter_parts = []
            for key, val in filters.items():
                filter_parts.append(f"{key} == '{val}'")
            filter_str = " and ".join(filter_parts)

        runs = api.runs(f"{entity}/{experiment}", filters=filter_str, per_page=limit)

        results = []
        for run in runs[:limit]:
            try:
                run_info = self.get_run(f"{entity}/{experiment}/{run.id}")
                results.append(run_info)
            except (ValueError, RuntimeError):
                continue

        return results


class MLflowTracker(ExperimentTracker):
    """MLflow experiment tracker integration."""

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        tracking_uri: str | None = None,
    ):
        """Initialize MLflow tracker.

        Args:
            api_key: MLflow API key (optional, mainly for Databricks).
            host: MLflow host URL.
            tracking_uri: MLflow tracking URI (overrides host).
        """
        super().__init__(api_key, host)
        self.tracking_uri = tracking_uri or host
        self._client = None

    def _get_client(self) -> Any:
        """Get or create MLflow client."""
        if self._client is None:
            try:
                import mlflow

                if self.tracking_uri:
                    mlflow.set_tracking_uri(self.tracking_uri)
                from mlflow.tracking import MlflowClient

                self._client = MlflowClient()
            except ImportError as e:
                raise RuntimeError("mlflow not installed. Install with: pip install mlflow") from e
        return self._client

    def get_run(self, run_id: str, **kwargs: Any) -> RunInfo:
        """Get a specific MLflow run by ID.

        Args:
            run_id: The run ID.
            **kwargs: Additional arguments (ignored).

        Returns:
            RunInfo with normalized run data.
        """
        client = self._get_client()
        run = client.get_run(run_id)

        data = run.data
        params = dict(data.params)
        metrics_dict = dict(data.metrics)
        tags = dict(run.data.tags)

        # Extract key metrics
        throughput = metrics_dict.get("throughput", metrics_dict.get("tok_per_sec", 0.0))
        latency = metrics_dict.get("latency", metrics_dict.get("mean_latency_ms", 0.0))
        memory = metrics_dict.get("gpu_memory_gb", metrics_dict.get("memory_gb", 0.0))

        # Get from params if not in metrics
        if throughput == 0.0:
            throughput = float(params.get("throughput", 0.0))
        if latency == 0.0:
            latency = float(params.get("latency", 0.0))
        if memory == 0.0:
            memory = float(params.get("memory_gb", 0.0))

        tokens = float(params.get("tokens_per_request", 1000))

        # Get experiment name
        experiment = client.get_experiment(run.info.experiment_id)
        experiment_name = experiment.name if experiment else "unknown"

        # Build URL
        tracking_uri = self.tracking_uri or ""
        url = f"{tracking_uri}/#/experiments/{run.info.experiment_id}/runs/{run_id}"

        return RunInfo(
            run_id=run_id,
            experiment=experiment_name,
            tracker="mlflow",
            parameters_b=self._extract_model_size(params),
            memory_gb=float(memory) if memory else 0.0,
            throughput_toks_per_sec=float(throughput) if throughput else 0.0,
            latency_ms_per_request=float(latency) if latency else 0.0,
            tokens_per_request=tokens,
            power_watts=metrics_dict.get("power_watts"),
            requests_per_day=int(params.get("requests_per_day", 0)) or None,
            metrics=metrics_dict,
            config=params,
            tags=[k for k, v in tags.items() if v == "true"],
            url=url,
        )

    def list_runs(
        self,
        experiment: str | None = None,
        filters: Mapping[str, Any] | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> list[RunInfo]:
        """List MLflow runs matching criteria.

        Args:
            experiment: Experiment name or ID.
            filters: Search filters (tracker-specific).
            limit: Maximum number of runs.
            **kwargs: Additional arguments including 'experiment_id'.

        Returns:
            List of RunInfo objects.
        """
        client = self._get_client()

        experiment_id = kwargs.get("experiment_id")
        if not experiment_id and experiment:
            # Try to find experiment by name
            exp = client.get_experiment_by_name(experiment)
            if exp:
                experiment_id = exp.experiment_id

        if not experiment_id:
            raise ValueError("experiment (name or ID) is required for mlflow")

        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filters.get("query") if filters else None,
            max_results=limit,
        )

        results = []
        for run in runs:
            try:
                run_info = self.get_run(run.info.run_id)
                results.append(run_info)
            except (ValueError, RuntimeError):
                continue

        return results


# Registry of available trackers
TRACKERS: dict[str, type[ExperimentTracker]] = {
    "wandb": WandbTracker,
    "mlflow": MLflowTracker,
}


def get_tracker(
    tracker_name: str,
    api_key: str | None = None,
    host: str | None = None,
    **kwargs: Any,
) -> ExperimentTracker:
    """Get an experiment tracker by name.

    Args:
        tracker_name: One of 'wandb', 'mlflow'.
        api_key: API key for authentication.
        host: Tracker host URL.
        **kwargs: Additional tracker-specific arguments.

    Returns:
        ExperimentTracker instance.

    Raises:
        ValueError: If tracker is not recognized.
    """
    if tracker_name not in TRACKERS:
        raise ValueError(f"Unknown tracker '{tracker_name}'. Available: {list(TRACKERS.keys())}")

    tracker_class = TRACKERS[tracker_name]
    return tracker_class(api_key=api_key, host=host, **kwargs)


def run_to_scenario(
    run_info: RunInfo,
    name: str,
    electricity_cost_per_kwh: float = 0.15,
    annual_hardware_cost_usd: float = 24000.0,
    one_time_project_cost_usd: float = 27000.0,
    requests_per_day: int | None = None,
) -> DeploymentScenario:
    """Convert RunInfo to a DeploymentScenario.

    Args:
        run_info: Parsed experiment run info.
        name: Scenario name.
        electricity_cost_per_kwh: Electricity cost per kWh.
        annual_hardware_cost_usd: Annual hardware cost.
        one_time_project_cost_usd: One-time optimization project cost.
        requests_per_day: Override for requests per day.

    Returns:
        DeploymentScenario populated from run data.
    """
    reqs_per_day = requests_per_day or run_info.requests_per_day or 50000

    # Estimate power if not provided
    power = run_info.power_watts or run_info.memory_gb * 10

    return DeploymentScenario(
        name=name,
        parameters_b=run_info.parameters_b or 7.0,
        memory_gb=run_info.memory_gb,
        throughput_toks_per_sec=run_info.throughput_toks_per_sec,
        power_watts=power,
        requests_per_day=reqs_per_day,
        tokens_per_request=int(run_info.tokens_per_request),
        electricity_cost_per_kwh=electricity_cost_per_kwh,
        annual_hardware_cost_usd=annual_hardware_cost_usd,
        one_time_project_cost_usd=one_time_project_cost_usd,
    )


def import_from_tracker(
    tracker_name: str,
    run_id: str | None = None,
    experiment: str | None = None,
    limit: int = 1,
    api_key: str | None = None,
    host: str | None = None,
    **kwargs: Any,
) -> list[RunInfo]:
    """Import runs from an experiment tracker.

    Args:
        tracker_name: Tracker type ('wandb' or 'mlflow').
        run_id: Specific run ID to fetch (optional).
        experiment: Experiment/project name (required for listing).
        limit: Maximum number of runs to return.
        api_key: API key for authentication.
        host: Tracker host URL.
        **kwargs: Additional tracker-specific arguments.

    Returns:
        List of RunInfo objects.

    Raises:
        ValueError: If neither run_id nor experiment is provided.
    """
    tracker = get_tracker(tracker_name, api_key=api_key, host=host, **kwargs)

    if run_id:
        return [tracker.get_run(run_id, **kwargs)]

    if experiment:
        return tracker.list_runs(experiment=experiment, limit=limit, **kwargs)

    raise ValueError("Either run_id or experiment must be provided")
