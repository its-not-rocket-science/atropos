# Formal API Specification: RL Core Components

This document defines strict interfaces for core RL environment components in `atropos.rl_env`.

## 1) Component contracts

### 1. Parser

**Interface**

```python
class Parser(Protocol):
    async def parse(self, raw_action: int) -> LineWorldParsedAction: ...
```

**Inputs**
- `raw_action: int` (bool explicitly rejected).

**Outputs**
- `LineWorldParsedAction` where `delta ∈ {-1, +1}`.

**Invariants**
- Parsed action is canonical and domain-valid.

---

### 2. Generator

**Interface**

```python
class Generator(Protocol):
    async def generate(
        self,
        state: LineWorldState,
        parsed_action: LineWorldParsedAction,
    ) -> LineWorldTransition: ...
```

**Inputs**
- `state: LineWorldState`
- `parsed_action: LineWorldParsedAction`

**Outputs**
- `LineWorldTransition`

**Invariants**
- `step_idx >= 1`
- `done == (reached_goal or out_of_steps)`
- `position_after - position_before == action`

---

### 3. RewardFunction

**Interface**

```python
class RewardFunction(Protocol):
    async def compute(self, transition: LineWorldTransition) -> LineWorldReward: ...
```

**Inputs**
- `transition: LineWorldTransition`

**Outputs**
- `LineWorldReward`

**Invariants**
- `components` keys are exactly `{"progress", "success_bonus", "timeout_penalty"}`
- `value == sum(components.values())`

---

### 4. TrajectoryBuilder

**Interface**

```python
class TrajectoryBuilder(Protocol):
    async def build_step(
        self,
        transition: LineWorldTransition,
        reward: LineWorldReward,
        introspection: LineWorldIntrospection,
    ) -> LineWorldStepRecord: ...

    def reset(self) -> None: ...
```

**Inputs**
- `transition: LineWorldTransition`
- `reward: LineWorldReward`
- `introspection: LineWorldIntrospection`

**Outputs**
- `LineWorldStepRecord`

**Invariants**
- `step_idx >= 1`
- per-episode step index monotonicity enforced by builder implementation.

---

### 5. Orchestrator

**Interface**

```python
@dataclass
class LineWorldOrchestrator:
    initial_state: LineWorldState
    parser: Parser
    generator: Generator
    reward_function: RewardFunction
    trajectory_builder: TrajectoryBuilder

    async def async_reset(self) -> LineWorldState: ...
    async def async_step(self, raw_action: int) -> LineWorldStepRecord: ...
```

**Inputs**
- Initial component instances and initial state.
- Runtime action input through `async_step(raw_action)`.

**Outputs**
- Reset state from `async_reset`.
- Step artifact record from `async_step`.

**Invariants**
- state object always satisfies `LineWorldState` invariants before stepping.
- all subcomponent outputs are validated before propagation.

## 2) Validation layer

The contract module provides runtime validators:

- `validate_state(...)`
- `validate_parsed_action(...)`
- `validate_transition(...)`
- `validate_reward(...)`
- `validate_step_record(...)`

These are called at orchestration boundaries and inside component implementations to fail fast with explicit errors.

## 3) Type enforcement

Type strictness is enforced by:

- `Protocol`-based interfaces for parser/generator/reward/builder.
- `Literal[-1, 1]` (`LineWorldAction`) for canonical action domain.
- `dataclass(..., slots=True)` models with constructor-time invariant checks.
- immutable/frozen DTOs where mutation is unsafe (`ParsedAction`, `Transition`, `Reward`, `StepRecord`).

## 4) Versioning strategy

Adopt **semantic interface versioning** for component contracts:

- **MAJOR**: Breaking changes to Protocol signatures, required fields, or invariants.
- **MINOR**: Backward-compatible additions (new optional metadata fields, additive introspection keys).
- **PATCH**: Bugfixes and tighter validation that do not change accepted/returned schema shape.

### Recommended implementation policy

1. Expose contract version in package metadata and docs (`API_CONTRACT_VERSION = "1.x.y"`).
2. For upcoming breaking changes, add compatibility shims for one minor cycle.
3. Document deprecations with target removal version.
4. Gate breaking behavior behind an explicit opt-in flag during migration windows.
