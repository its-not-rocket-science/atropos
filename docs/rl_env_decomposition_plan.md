# Async RL Environment Decomposition Plan (LLM-Focused)

## Objective
Reduce cognitive complexity in a large base environment class **without reducing capability** and without a greenfield rewrite.

---

## 1) Responsibilities currently co-located in a monolithic `BaseEnvironment`

In async RL-for-LLM frameworks, the base env often accumulates these concerns:

1. **Episode lifecycle orchestration**
   - `reset()`, `step()`, termination, truncation, timeout logic.
2. **LLM generation / inference transport**
   - Prompt construction, model adapter calls, retry/backoff, streaming handling.
3. **Action parsing & validation**
   - Convert raw text/tool outputs into typed actions; schema checks; repair/fallback.
4. **State transition / world simulation**
   - Apply validated action to environment state; compute next observation payload.
5. **Reward/scoring computation**
   - Sparse/dense rewards, penalties, shaping, normalization, clipping.
6. **Instrumentation and logging**
   - Traces, per-step metrics, token usage, debug snapshots.
7. **Persistence / replay export**
   - Trajectory serialization, artifacts, failure dumps.
8. **Concurrency controls**
   - Async task scheduling, cancellation, timeouts, semaphore limits.
9. **Configuration plumbing**
   - Merge defaults/overrides, experiment flags, runtime adapters.

This leads to high fan-in/out, difficult test seams, and brittle extension points.

---

## 2) Proposed decomposition (single-responsibility components + interfaces)

### Core design principle
Keep a **thin compatibility facade** (`BaseEnvironment`) that delegates to explicit collaborators.

### Components and boundaries

#### A. `EpisodeEngine`
**Responsibility**: orchestrate reset/step/terminal flow only.

**Interface**
- `async reset(seed: int | None = None) -> Observation`
- `async step(action_input: AgentOutput) -> StepResult`
- `async close() -> None`

Depends on: `ActionPipeline`, `TransitionModel`, `RewardModel`, `TelemetrySink`, `TrajectoryStore`.

---

#### B. `GenerationClient`
**Responsibility**: model/tool invocation and transport concerns.

**Interface**
- `async generate(request: GenerationRequest) -> GenerationResponse`

Handles retry, timeout, streaming aggregation, and provider-specific adaptation.

---

#### C. `ActionPipeline`
**Responsibility**: parse, validate, and normalize generated output into domain action.

**Interface**
- `parse(raw: str) -> ParsedAction`
- `validate(parsed: ParsedAction, state: EnvState) -> ValidationResult`
- `normalize(parsed: ParsedAction) -> DomainAction`

---

#### D. `TransitionModel`
**Responsibility**: pure state transition logic.

**Interface**
- `apply(state: EnvState, action: DomainAction) -> TransitionOutcome`

No logging and no reward logic inside.

---

#### E. `RewardModel`
**Responsibility**: compute reward/aux metrics from transition outcome.

**Interface**
- `compute(prev_state: EnvState, outcome: TransitionOutcome) -> RewardResult`

Can be composed as: `CompositeRewardModel([task_reward, safety_penalty, format_bonus])`.

---

#### F. `ObservationBuilder`
**Responsibility**: construct agent-visible observation from internal state/outcome.

**Interface**
- `build(state: EnvState, outcome: TransitionOutcome | None) -> Observation`

---

#### G. `TelemetrySink`
**Responsibility**: structured metrics/events/traces only.

**Interface**
- `on_step(event: StepTelemetry) -> None`
- `on_error(event: ErrorTelemetry) -> None`
- `flush() -> None`

Adapters: stdout/jsonl/OpenTelemetry/W&B.

---

#### H. `TrajectoryStore`
**Responsibility**: persist and retrieve trajectories/checkpoints.

**Interface**
- `append(record: TrajectoryRecord) -> None`
- `finalize(episode_id: str) -> None`

---

#### I. `AsyncRuntimePolicy`
**Responsibility**: execution policy for concurrency.

**Interface**
- `run_with_timeout(coro, timeout_s: float)`
- `guarded_gather(tasks, max_concurrency: int)`
- `cancel_scope(task_ids: list[str])`

---

## 3) Incremental migration plan (backward compatible)

### Phase 0: Stabilize contract (no behavior change)
1. Freeze and document current `BaseEnvironment` public API.
2. Add golden trajectory tests around representative envs.
3. Add step-level snapshot tests for reward and termination behavior.

### Phase 1: Extract passive collaborators first
1. Move parsing logic into `ActionPipeline`.
2. Move reward logic into `RewardModel`.
3. Move logging into `TelemetrySink`.
4. Keep `BaseEnvironment` calling old method names but delegate internally.

### Phase 2: Extract orchestration
1. Introduce `EpisodeEngine` used by `BaseEnvironment.step/reset`.
2. Keep legacy hooks (`before_step`, `after_step`, etc.) via adapter bridge.

### Phase 3: Extract async/runtime layer
1. Move timeout/retry/concurrency controls into `AsyncRuntimePolicy` + `GenerationClient`.
2. Enable feature flag for new runtime path.

### Phase 4: Promote new extension points
1. Mark monolithic overrides as deprecated (with timeline).
2. Provide migration guide mapping old hooks -> new interfaces.
3. Add compatibility shim for at least 2 minor releases.

### Backward-compatibility mechanics
- `BaseEnvironment` remains import-stable and method-stable.
- Legacy subclass methods are still invoked through `LegacyHookAdapter`.
- Config flag:
  - `env.architecture_mode = "legacy" | "hybrid" | "componentized"`.

---

## 4) BEFORE vs AFTER architecture diagrams

### BEFORE (monolithic)

```text
+-------------------------------------------------------------+
|                       BaseEnvironment                       |
|-------------------------------------------------------------|
| reset/step lifecycle                                        |
| prompt/generation transport                                 |
| parsing + validation                                        |
| transition logic                                            |
| reward computation                                          |
| logging/metrics/tracing                                     |
| persistence/replay                                          |
| async timeout/retry/concurrency                             |
| config plumbing                                             |
+-------------------------------------------------------------+
```

### AFTER (componentized + compatibility facade)

```text
           +------------------------------+
           |        BaseEnvironment       |
           |  (stable facade + shim)      |
           +--------------+---------------+
                          |
                          v
                 +------------------+
                 |   EpisodeEngine  |
                 +--+----+----+----+
                    |    |    |    |
                    |    |    |    +--------------------+
                    |    |    +-----> TelemetrySink     |
                    |    |         +--------------------+
                    |    +---------> RewardModel        |
                    |              +--------------------+
                    +-------------> ActionPipeline      |
                    |              +--------------------+
                    +-------------> TransitionModel     |
                    |              +--------------------+
                    +-------------> ObservationBuilder  |
                    |              +--------------------+
                    +-------------> TrajectoryStore     |
                    |
                    +-------------> GenerationClient
                                   (uses AsyncRuntimePolicy)
```

---

## 5) Risks of over-modularization (and controls)

1. **Interface explosion / ceremony**
   - Mitigation: keep ~6–9 core interfaces; avoid micro-classes for trivial transforms.
2. **Performance overhead from indirection**
   - Mitigation: batch telemetry writes; avoid per-token deep adapter chains in hot path.
3. **Debugging across many boundaries**
   - Mitigation: shared correlation IDs (`run_id`, `episode_id`, `step_id`) and unified step trace.
4. **Inconsistent ownership of state**
   - Mitigation: single authoritative `EnvState` lifecycle owned by `EpisodeEngine`.
5. **Partial migration drift (legacy + new behavior mismatch)**
   - Mitigation: golden trajectory parity tests and feature-flag A/B validation.

---

## Suggested file structure

```text
src/rl_env/
  base_environment.py            # compatibility facade
  engine/
    episode_engine.py
    runtime_policy.py
  generation/
    client.py
    request_response.py
  action/
    pipeline.py
    parsers.py
    validators.py
  transition/
    model.py
    state.py
  reward/
    model.py
    composite.py
  observation/
    builder.py
  telemetry/
    sink.py
    events.py
  trajectory/
    store.py
    records.py
  compat/
    legacy_hook_adapter.py
```

---

## Example refactored class skeletons

```python
# src/rl_env/base_environment.py
from __future__ import annotations

class BaseEnvironment:
    """Backward-compatible facade. Public API remains stable."""

    def __init__(self, engine, legacy_adapter=None):
        self._engine = engine
        self._legacy = legacy_adapter

    async def reset(self, seed: int | None = None):
        if self._legacy:
            self._legacy.before_reset(seed)
        obs = await self._engine.reset(seed=seed)
        if self._legacy:
            self._legacy.after_reset(obs)
        return obs

    async def step(self, action_input):
        if self._legacy:
            self._legacy.before_step(action_input)
        result = await self._engine.step(action_input)
        if self._legacy:
            self._legacy.after_step(result)
        return result
```

```python
# src/rl_env/engine/episode_engine.py
from __future__ import annotations

class EpisodeEngine:
    def __init__(
        self,
        generation_client,
        action_pipeline,
        transition_model,
        reward_model,
        observation_builder,
        telemetry_sink,
        trajectory_store,
    ):
        self.generation_client = generation_client
        self.action_pipeline = action_pipeline
        self.transition_model = transition_model
        self.reward_model = reward_model
        self.observation_builder = observation_builder
        self.telemetry_sink = telemetry_sink
        self.trajectory_store = trajectory_store
        self.state = None

    async def reset(self, seed: int | None = None):
        self.state = self.transition_model.initial_state(seed=seed)
        obs = self.observation_builder.build(self.state, outcome=None)
        return obs

    async def step(self, action_input):
        parsed = self.action_pipeline.parse(action_input.raw_text)
        valid = self.action_pipeline.validate(parsed, self.state)
        action = self.action_pipeline.normalize(valid.value)

        outcome = self.transition_model.apply(self.state, action)
        reward = self.reward_model.compute(self.state, outcome)
        self.state = outcome.next_state

        obs = self.observation_builder.build(self.state, outcome)
        result = {
            "observation": obs,
            "reward": reward.value,
            "done": outcome.done,
            "info": {"metrics": reward.metrics},
        }
        self.telemetry_sink.on_step({"result": result})
        self.trajectory_store.append({"action": action, "outcome": outcome})
        return result
```

```python
# src/rl_env/action/pipeline.py
from __future__ import annotations

class ActionPipeline:
    def __init__(self, parser, validator, normalizer):
        self.parser = parser
        self.validator = validator
        self.normalizer = normalizer

    def parse(self, raw: str):
        return self.parser.parse(raw)

    def validate(self, parsed, state):
        return self.validator.validate(parsed, state)

    def normalize(self, parsed):
        return self.normalizer.normalize(parsed)
```
