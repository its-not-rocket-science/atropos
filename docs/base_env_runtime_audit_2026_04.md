# BaseEnv Runtime Audit (April 2026)

## Scope

This audit reviews `src/atroposlib/envs/base.py` and runtime-related modules under
`src/atroposlib/envs/` to ensure `BaseEnv` remains a thin contract + orchestration
surface instead of a god class.

## 1) Responsibilities still owned by `BaseEnv` (before this change)

`BaseEnv` was already significantly slimmed down, but still retained a few mixed concerns:

1. **Dependency wiring + lifecycle bootstrap**
   - Chose defaults and resolved legacy constructor aliases (`worker_manager`, `logging_manager`, `transport`).
   - Built runtime controller and env-logic adapters inline.
   - Managed seed bootstrapping and propagation.
2. **Compatibility method implementations**
   - Implemented inline wrappers for serving streams, reset behavior, CLI conversion,
     config merge, worker orchestration alias, transport alias, logging alias, and checkpoint alias.
3. **Environment contract/orchestration API (kept intentionally)**
   - `step`, `process`, `evaluate`, `serve`, and compatibility properties.

## 2) Extraction performed

To remove remaining god-class behavior while preserving public ergonomics:

- Added **`dependency_factory.py`**:
  - `BaseEnvDependencyFactory` now owns collaborator construction, legacy alias resolution,
    runtime controller composition, and seed propagation.
  - Emits a `RuntimeWiring` DTO with explicit dependencies.
- Added **`compatibility_adapter.py`**:
  - `BaseEnvCompatibilityAdapter` now owns legacy wrappers and utility compatibility calls:
    stream serving, reset, CLI/config adapters, and legacy alias methods.
- Updated **`BaseEnv`** to:
  - call factory for composition,
  - delegate compatibility behavior to adapter,
  - retain stable public method/property surface.

## 3) Updated runtime module dependency diagram

```mermaid
flowchart TD
  BaseEnv[BaseEnv\n(thin orchestration facade)]
  Factory[BaseEnvDependencyFactory]
  Compat[BaseEnvCompatibilityAdapter]
  Controller[RuntimeController]

  EnvLogic[EnvLogic / PassthroughEnvLogic]
  WorkerRuntime[WorkerRuntime]
  TransportClient[TransportClient]
  MetricsLogger[MetricsLogger]
  CheckpointManager[CheckpointManager]
  CliAdapter[CliAdapter]

  BaseEnv --> Factory
  BaseEnv --> Compat
  BaseEnv --> Controller

  Factory --> WorkerRuntime
  Factory --> TransportClient
  Factory --> MetricsLogger
  Factory --> CheckpointManager
  Factory --> CliAdapter
  Factory --> EnvLogic
  Factory --> Controller

  Controller --> WorkerRuntime
  Controller --> TransportClient
  Controller --> MetricsLogger
  Controller --> CheckpointManager
  Controller --> EnvLogic

  Compat --> WorkerRuntime
  Compat --> TransportClient
  Compat --> MetricsLogger
  Compat --> CheckpointManager
  Compat --> CliAdapter
```

## 4) Testability decomposition summary

Each extracted unit has isolated tests:

- `BaseEnvDependencyFactory` wiring + seed propagation contract.
- `BaseEnvCompatibilityAdapter` stream/single serve behavior.
- `BaseEnvCompatibilityAdapter` delegation of legacy APIs.

This keeps `BaseEnv` tests focused on external behavior while unit tests verify
internal composition seams.

## 5) BaseEnv complexity summary (before vs after)

Measured with a lightweight AST script:

| Metric | Before | After | Delta |
|---|---:|---:|---:|
| File LOC (`base.py`) | 158 | 159 | +1 |
| `BaseEnv` methods | 17 | 17 | 0 |
| Class decision nodes (`if/for/while/try/...`) | 10 | 1 | -9 |

Interpretation:
- Public API count is unchanged (backward compatibility preserved).
- Conditional/wiring complexity moved out of `BaseEnv` into dedicated, testable services.

## Backward compatibility status

Preserved:

- Constructor aliases (`worker_manager`, `transport`, `logging_manager`).
- Legacy methods (`orchestrate_workers`, `call_api`, `log_event`, `checkpoint`).
- Orchestration aliases (`process`, `evaluate`, `serve`).
- Legacy collaborator properties (`runtime`, `worker_manager`, `transport`, `logger`, `logging_manager`).
