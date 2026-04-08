# RL Environment Component Contracts

This module split separates environment concerns into explicit, independently testable units while preserving original `LineWorldEnv` behavior.

## New module structure

```text
src/atropos/rl_env/
  __init__.py
  contracts.py
  line_world.py
examples/minimal_trajectory_example.py
```

## Component definitions

### 1) `Parser`

- **Input**: `raw_action: int`.
- **Output**: `LineWorldParsedAction(delta: int)`.
- **Failure modes**:
  - `TypeError` if input is not `int`.
  - `ValueError` if input is outside action space `{-1, +1}`.
- **Test strategy**:
  - unit test valid action mapping.
  - unit test invalid type/value rejections.

### 2) `Generator`

- **Input**: `(state: LineWorldState, parsed_action: LineWorldParsedAction)`.
- **Output**: `LineWorldTransition` (state delta + terminal flags).
- **Failure modes**:
  - `RuntimeError` when state invariants fail (`max_steps <= 0`).
  - `ValueError`/`RuntimeError` for malformed parsed action or impossible transitions.
- **Test strategy**:
  - deterministic transition test with fixed start state.
  - terminal-condition tests (`reached_goal`, `out_of_steps`, `done`).

### 3) `RewardFunction`

- **Input**: `LineWorldTransition`.
- **Output**: `LineWorldReward(value, components)`.
- **Failure modes**:
  - `RuntimeError` for transition invariant violations.
- **Test strategy**:
  - unit tests for progress-only, success bonus, timeout penalty, and composed totals.

### 4) `TrajectoryBuilder`

- **Input**: `(transition, reward)`.
- **Output**: `LineWorldStepRecord`.
- **Failure modes**:
  - `RuntimeError` when step ordering is non-monotonic.
- **Test strategy**:
  - append-order tests.
  - reset test to verify episode boundary semantics.

### 5) `EnvOrchestrator` (`LineWorldOrchestrator`)

- **Input**: component instances + initial state.
- **Output**:
  - `async_reset() -> LineWorldState`.
  - `async_step(raw_action) -> LineWorldStepRecord`.
- **Failure modes**:
  - propagates collaborator exceptions without swallowing.
  - state mismatch bugs manifest as transition/reward invariant exceptions.
- **Test strategy**:
  - integration test asserting equivalence to legacy sync behavior.
  - async tests validating `async_step`/`async_reset` flow.

## Refactored real environment: `LineWorldEnv`

`LineWorldEnv` is now a thin compatibility wrapper over `LineWorldOrchestrator`:

- Keeps original sync methods (`reset`, `step`) for no-break migration.
- Adds async methods (`async_reset`, `async_step`) for concurrent rollouts.
- Preserves reward and done semantics from the original example implementation.
