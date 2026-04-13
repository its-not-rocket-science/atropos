# BaseEnv Refactor Migration Notes

This document describes the extraction of `BaseEnv` responsibilities into focused components while preserving runtime behavior.

## What changed

`BaseEnv` now acts as a thin orchestration layer and environment contract, with dependency-injected collaborators:

- `WorkerRuntime` (worker orchestration and dynamic scaling)
- `TransportClient` (API transport boundary)
- `CheckpointManager` (snapshot persistence)
- `MetricsLogger` (events + metrics)
- `CliAdapter` (CLI/config interoperability)

## Compatibility shims

To avoid breaking existing environments, the previous symbols remain available:

- `WorkerManager` now subclasses `WorkerRuntime`.
- `LoggingManager` now subclasses `MetricsLogger`.
- `BaseEnv` constructor still accepts legacy names like `worker_manager` and `logging_manager`.
- `BaseEnv` still exposes compatibility properties (`runtime`, `worker_manager`, `logger`, `logging_manager`, `transport`).
- `BaseEnv` provides `serve`, `process`, and `evaluate` as compatibility aliases around the same orchestration path used by `step`.

## New module layout

- `src/atroposlib/envs/worker_runtime.py`
- `src/atroposlib/envs/transport_client.py`
- `src/atroposlib/envs/checkpoint_manager.py`
- `src/atroposlib/envs/metrics_logger.py`
- `src/atroposlib/envs/cli_adapter.py`
- `src/atroposlib/envs/base.py`

Legacy shim modules retained:

- `src/atroposlib/envs/worker_manager.py`
- `src/atroposlib/envs/logging_manager.py`

## Migration guidance

1. New code should depend on `WorkerRuntime`, `MetricsLogger`, and `CliAdapter` directly.
2. Existing code using `WorkerManager` / `LoggingManager` can migrate gradually with no immediate behavior change.
3. Prefer constructor injection in custom envs:

```python
env = BaseEnv(
    worker_runtime=WorkerRuntime(max_workers=16),
    transport_client=MyTransportClient(),
    metrics_logger=MetricsLogger(),
    checkpoint_manager=CheckpointManager(),
    cli_adapter=CliAdapter(),
)
```

This keeps orchestration policy explicit and easier to harden in production.
