# Trajectory Data Pipeline Analysis

This document maps the current codebase into an explicit, debuggable data pipeline for trajectory-like experiment flow.

## 1) Lifecycle trace: input → generation → parsing → scoring → storage

### As implemented today (A/B testing path)

```text
[ABTestConfig + deployed variant endpoints]
  -> ExperimentRunner.start()
  -> background monitoring loop

input
  -> _collect_telemetry_metrics() (real) OR _collect_metrics() (synthetic fallback)

generation
  -> telemetry collector returns TelemetryData (aggregated + samples)
  -> runner builds VariantMetrics per variant

parsing/normalization
  -> metric-name mapping (telemetry_field_map)
  -> convert values to {mean,std,count}

scoring
  -> analyze_experiment_results()
  -> synthetic sample reconstruction from aggregate stats
  -> analyze_variant_comparison() produces p-value/effect size/significance

storage/output
  -> ExperimentResult assembled in _get_current_result()
  -> ExperimentStore.save_result()/update_experiment() JSON persistence
```

### Planned RL-style trajectory path (from decomposition plan)

```text
AgentOutput.raw_text
  -> ActionPipeline.parse()
  -> ActionPipeline.validate(state)
  -> ActionPipeline.normalize()
  -> TransitionModel.apply(state, action)
  -> RewardModel.compute(prev_state, outcome)
  -> EpisodeEngine emits result + telemetry
  -> TrajectoryStore.append({action, outcome})
```

## 2) Where state is implicit or mutated

1. **Runner-global mutable caches**
   - `ExperimentRunner` mutates `_variant_metrics`, `_statistical_results`, `_deployment_ids`, `_error_counts`, `_status`, `_start_time`, `_end_time` across async thread cycles.
   - This makes replay/debugging hard because each cycle overwrites prior snapshots.

2. **Dual-mode metric source is implicit**
   - `_collect_metrics()` silently falls back to synthetic random metrics if telemetry collection fails.
   - This can contaminate scoring with non-production-like data unless explicitly marked.

3. **Scoring reconstructs pseudo-raw samples**
   - `analyze_experiment_results()` generates Gaussian samples from summary statistics.
   - The synthetic reconstruction is not carried forward as lineage metadata, so downstream consumers cannot distinguish "real" vs "simulated" evidence.

4. **Storage payload schema is weakly typed at write boundary**
   - `ExperimentStore` persists generic dicts (`save_experiment`) and mixes config/result/status fields.
   - Status is stored as enum `.value` (int) in some paths, while read/list logic treats status as lower-case strings.

5. **Trajectory contract only exists in planning doc**
   - The RL decomposition defines `TrajectoryStore.append(record)` but record type is not yet concrete, inviting drift between parser/reward/output schemas.

## 3) Proposed typed schemas

Use pydantic v2 models (or frozen dataclasses with validators). Pydantic shown below for validation and JSON-schema export.

```python
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, ConfigDict


class RewardComponent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RewardOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float
    components: list[RewardComponent] = Field(default_factory=list)
    penalties: dict[str, float] = Field(default_factory=dict)
    normalized: float | None = None
    uncertainty: float | None = None
    scoring_version: str


class StepRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_idx: int
    timestamp: datetime
    input_payload: dict[str, Any]
    generated_text: str | None = None
    parsed_action: dict[str, Any] | None = None
    transition_outcome: dict[str, Any]
    reward: RewardOutput
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)


class GroupType(str, Enum):
    CONTROL = "control"
    TREATMENT = "treatment"


class GroupRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str
    experiment_id: str
    group_type: GroupType
    variant_ids: list[str]
    traffic_weight: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trajectory_id: str
    episode_id: str
    experiment_id: str | None = None
    group_id: str | None = None
    source: Literal["production", "replay", "synthetic"]
    start_time: datetime
    end_time: datetime | None = None
    initial_state: dict[str, Any]
    final_state: dict[str, Any] | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

## 4) Logging hooks to make flow debuggable

Emit structured events with stable keys (`event_name`, `experiment_id`, `trajectory_id`, `step_idx`, `stage_ms`, `source`).

1. **Input hook** (`input.received`)
   - At monitor cycle start and per variant endpoint pull.
   - Include endpoint, server_type, requested metrics, polling window.

2. **Generation hook** (`metrics.collected`)
   - Capture collector success/failure, sample_count, latency of collection, fallback reason.

3. **Parsing hook** (`metrics.parsed`)
   - Emit field mapping decisions (requested metric -> telemetry field), dropped metrics, type coercions.

4. **Scoring hook** (`scoring.completed`)
   - Emit control/treatment IDs, test type, p-value, effect size, significance, whether synthetic samples used.

5. **Storage hook** (`storage.write`)
   - Emit path, bytes written, schema version, checksum, write duration, atomic replace status.

6. **Trajectory step hook** (`trajectory.step`)
   - For RL path: include parsed action hash, reward total + components, done flag, state hash before/after.

## 5) Refactoring suggestions for independent testability

1. **Split collection into pure + side-effect layers**
   - `TelemetryFetcher.fetch(endpoint)->RawTelemetryBlob`
   - `TelemetryParser.parse(blob)->ParsedMetrics`
   - `MetricsAggregator.aggregate(parsed)->VariantMetrics`
   - Unit-test each with deterministic fixtures.

2. **Make scoring purely functional**
   - `score_variant_pair(control: MetricSeries, treatment: MetricSeries, config)->StatisticalResult`
   - Inject RNG or remove synthetic sampling by requiring raw samples in typed input.

3. **Introduce explicit pipeline context object**
   - Immutable `PipelineContext` passed stage-to-stage (instead of mutating runner fields).
   - Keep append-only `stage_events` for replay and debugging.

4. **Version storage schema and enforce at boundary**
   - `ExperimentEnvelope{schema_version, payload_type, payload}`.
   - Validate before write and on read; reject or migrate old versions.

5. **Add stage contract tests**
   - Golden tests for: telemetry parse mapping, scoring parity, storage round-trip.
   - Add trajectory-level snapshot test that verifies full lineage (`input -> parse -> score -> store`) with fixed seed.

6. **Threading determinism**
   - Move monitor loop clock/sleep behind interface (`Clock`, `Sleeper`) and inject fake time in tests.
   - Expose one-cycle method (`run_monitor_cycle`) to test exactly one pipeline pass without background threads.

## Suggested target diagram (explicit contracts)

```text
RawInput
  -> [InputAdapter] -> InputEvent
  -> [Collector] -> CollectedMetrics
  -> [Parser/Normalizer] -> ParsedMetrics
  -> [Scorer] -> RewardOutput/StatisticalResult
  -> [TrajectoryAssembler] -> TrajectoryRecord
  -> [Store] -> PersistedEnvelope

At every edge: typed model + structured log + stage timing + trace_id
```
