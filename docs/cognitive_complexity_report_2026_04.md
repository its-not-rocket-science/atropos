# Cognitive Complexity Report (April 2026)

## Scope and method

Analyzed the Python codebase under `src/atropos` with a lightweight static AST scan to estimate:

- file size (lines of text),
- class overload risk (composite score = `2 * method_count + branch_nodes`),
- responsibility breadth (count of distinct method-intent buckets inferred from method names).

This is not a strict Sonar-style cognitive complexity score, but a practical architecture triage metric for identifying where refactoring likely yields the largest maintainability gains.

## 1) Largest files

Top files by line count (repository-wide):

| Rank | File | Lines |
|---|---|---:|
| 1 | `src/atropos/cli.py` | 2402 |
| 2 | `scripts/patched_prune.py` | 1471 |
| 3 | `src/atropos/visualization.py` | 1126 |
| 4 | `src/atropos/pruning_integration.py` | 994 |
| 5 | `scripts/measure_quality_speed_tradeoffs.py` | 926 |
| 6 | `src/atropos/abtesting/runner.py` | 798 |
| 7 | `src/atropos/deployment/platforms.py` | 781 |
| 8 | `scripts/combined_optimization.py` | 748 |
| 9 | `scripts/validate_on_models.py` | 716 |
| 10 | `src/atropos/pipeline/runner.py` | 690 |

Primary maintainability hotspots in production package code (`src/atropos`) are therefore:

- `cli.py`
- `visualization.py`
- `pruning_integration.py`
- `abtesting/runner.py`
- `deployment/platforms.py`
- `pipeline/runner.py`

## 2) Most overloaded classes

Top overloaded classes by composite overload score (`2 * methods + branches`):

| Rank | Class | File | Methods | Branch nodes | Responsibility buckets | Overload score |
|---|---|---|---:|---:|---:|---:|
| 1 | `ExperimentRunner` | `src/atropos/abtesting/runner.py` | 20 | 90 | 10 | 130 |
| 2 | `PipelineRunner` | `src/atropos/pipeline/runner.py` | 10 | 54 | 2 | 74 |
| 3 | `HyperparameterTuner` | `src/atropos/tuning/hyperparameter_tuner.py` | 13 | 34 | 2 | 60 |
| 4 | `CloudPricingEngine` | `src/atropos/costs/cloud_pricing.py` | 12 | 34 | 5 | 58 |
| 5 | `ModelValidator` | `src/atropos/validation/runner.py` | 11 | 27 | 3 | 49 |
| 6 | `DistributedBenchmarkWrapper` | `src/atropos/validation/distributed_benchmark.py` | 5 | 32 | 3 | 42 |
| 7 | `PruningFramework` | `src/atropos/pruning/base.py` | 11 | 17 | 6 | 39 |
| 8 | `VLLMCollector` | `src/atropos/telemetry_collector.py` | 7 | 25 | 3 | 39 |
| 9 | `ExperimentStore` | `src/atropos/abtesting/store.py` | 9 | 20 | 6 | 38 |
| 10 | `WandbTracker` | `src/atropos/integrations.py` | 4 | 27 | 3 | 35 |

### Responsibility breadth observations

- **Very high breadth + very high branching:** `ExperimentRunner` is the most urgent target.
- **High branching with low bucket diversity:** `PipelineRunner` and `HyperparameterTuner` look control-flow heavy rather than domain-broad; simplify decision logic and orchestration paths.
- **Moderate breadth with infra concerns mixed in:** `CloudPricingEngine`, `ExperimentStore`, and `PruningFramework` combine multiple concerns and can be split along data access/strategy boundaries.

## 3) Responsibilities per class (quantified)

Responsibility buckets for top classes (derived from method naming intent):

| Class | Responsibility buckets (count) | Interpretation |
|---|---:|---|
| `ExperimentRunner` | 10 | lifecycle control, analysis, metrics/reporting, deployment-adjacent flow, state transitions |
| `PipelineRunner` | 2 | orchestration + execution path branching (overly centralized control flow) |
| `HyperparameterTuner` | 2 | search/tune core but with high internal logic depth |
| `CloudPricingEngine` | 5 | pricing retrieval, defaults, estimation, catalog listing, data refresh |
| `ModelValidator` | 3 | validation orchestration, checks, reporting/aggregation |
| `DistributedBenchmarkWrapper` | 3 | benchmark orchestration, distributed execution, result normalization |
| `PruningFramework` | 6 | framework adaptation, prune/test/build/introspection concerns mixed |
| `VLLMCollector` | 3 | collection orchestration, extraction, transport/storage |
| `ExperimentStore` | 6 | CRUD + indexing/listing + persistence policy |
| `WandbTracker` | 3 | run lifecycle + logging + metadata sync |

## 4) Simplification proposals

### A. `ExperimentRunner` (highest priority)

**Problems**
- Mixes experiment lifecycle, variant selection, statistics, gating decisions, and persistence interactions.
- Branch-heavy methods increase regression risk.

**Simplifications**
- Extract a pure `ExperimentDecisionEngine` for stop/pause/resume/winner selection logic.
- Extract `ExperimentExecutionService` for run lifecycle mechanics.
- Keep `ExperimentRunner` as thin facade coordinating injected collaborators.
- Replace nested conditionals with rule objects (strategy pattern) for experiment state transitions.

### B. `PipelineRunner`

**Problems**
- Centralized orchestration with heavy conditional dispatch.

**Simplifications**
- Split into `PipelinePlanner` (build plan DAG) and `PipelineExecutor` (execute nodes).
- Introduce typed stage handlers (`ValidationStage`, `PruningStage`, `ReportingStage`) instead of large `if/elif` chains.

### C. `HyperparameterTuner`

**Problems**
- Search policy, objective evaluation, and result tracking likely co-located.

**Simplifications**
- Separate into `SearchStrategy`, `ObjectiveEvaluator`, and `TuningHistoryStore`.
- Make objective evaluation side-effect free for testability.

### D. `CloudPricingEngine`

**Problems**
- Provider normalization, cache refresh, and estimation math likely intertwined.

**Simplifications**
- Provider adapters per cloud (`AWSPricingAdapter`, `AzurePricingAdapter`, `GCPPricingAdapter`).
- `PricingCatalog` for normalized data access.
- `CostEstimator` for deterministic math (no I/O).

### E. `PruningFramework` + integrations

**Problems**
- Base abstraction includes too many concerns (capabilities, invocation semantics, validation/test hooks).

**Simplifications**
- Keep `PruningFramework` minimal (protocol + metadata).
- Move execution details into framework-specific `Runner` classes.
- Separate capability discovery from execution path.

## 5) Proposed class splits (concrete)

### `ExperimentRunner` split
- `ExperimentRunner` (facade)
- `ExperimentExecutionService`
- `ExperimentDecisionEngine`
- `ExperimentMetricsAggregator`
- `ExperimentStateRepository` (via store abstraction)

### `PipelineRunner` split
- `PipelineRunner` (facade)
- `PipelinePlanner`
- `PipelineExecutor`
- `StageHandler` implementations per domain step

### `CloudPricingEngine` split
- `CloudPricingEngine` (API facade)
- `PricingCatalog` (read/caching)
- `CostEstimator` (math)
- `ProviderAdapter` implementations

## 6) Prioritized refactor plan

### Phase 0 — Baseline safety (1–2 days)
1. Freeze current behavior with characterization tests around A/B runner and pipeline runner outputs.
2. Add lightweight complexity guardrail script to CI (method count / branch thresholds per class).

### Phase 1 — Highest ROI decomposition (3–5 days)
1. Refactor `ExperimentRunner` into facade + extracted decision/execution services.
2. Introduce dependency injection seams for statistics and persistence collaborators.
3. Keep external API backward-compatible.

### Phase 2 — Orchestration untangling (3–4 days)
1. Split `PipelineRunner` into planner/executor/stage handlers.
2. Replace large branching with registry-based stage dispatch.

### Phase 3 — Pricing and integration cleanup (3–5 days)
1. Extract `CostEstimator` and provider adapters from `CloudPricingEngine`.
2. Slim `PruningFramework` base and move behavior into framework-specific runner classes.

### Phase 4 — Hardening and governance (ongoing)
1. Enforce complexity budgets in CI (e.g., max methods/class, max branch nodes/method).
2. Track trend of top-10 overloaded classes per release.
3. Require architecture notes for new classes exceeding threshold.

## Suggested success metrics

- 30–40% reduction in overload score for `ExperimentRunner` and `PipelineRunner`.
- No class in `src/atropos` with >12 methods **and** >40 branch nodes.
- Reduced PR review time and defect rate in orchestration modules.
