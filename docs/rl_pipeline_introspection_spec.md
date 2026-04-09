# RL Pipeline Introspection Spec

## Goal
Enable users to answer: **"why did this trajectory get this reward?"** by attaching structured introspection to every trajectory step.

## Stage Model
Each step carries stage-level introspection for:
- `generation`
- `parsing`
- `scoring`

Each stage contains:
1. `metadata` (versioning, component identity, static context)
2. `intermediate_output` (machine-parseable values from that stage)
3. `reasoning_trace` (human-readable ordered explanation)

## Step Introspection Schema
```json
{
  "step_idx": 5,
  "action": 1,
  "position_before": 4,
  "position_after": 5,
  "reward": 11.0,
  "done": true,
  "introspection": {
    "generation": {
      "stage": "generation",
      "metadata": {
        "generator": "LineWorldGenerator",
        "goal": 5,
        "max_steps": 20
      },
      "intermediate_output": {
        "position_before": 4,
        "position_after": 5,
        "step_idx": 5,
        "reached_goal": true,
        "out_of_steps": false
      },
      "reasoning_trace": [
        "Applied delta=1 to position=4.",
        "New position=5, step_idx=5.",
        "Computed done flag from reached_goal OR out_of_steps."
      ]
    },
    "parsing": {
      "stage": "parsing",
      "metadata": {
        "parser": "LineWorldParser",
        "action_space": "[-1, +1]"
      },
      "intermediate_output": {
        "raw_action": 1,
        "parsed_delta": 1
      },
      "reasoning_trace": [
        "Received raw action 1.",
        "Validated action against [-1, +1] and mapped to delta=1."
      ]
    },
    "scoring": {
      "stage": "scoring",
      "metadata": {
        "reward_model": "LineWorldRewardFunction",
        "components": "progress,success_bonus,timeout_penalty"
      },
      "intermediate_output": {
        "progress": 1.0,
        "success_bonus": 10.0,
        "timeout_penalty": 0.0,
        "total_reward": 11.0
      },
      "reasoning_trace": [
        "progress = position_after - position_before.",
        "success_bonus = 10.0 when reached_goal else 0.0.",
        "timeout_penalty = -5.0 when out_of_steps and not reached_goal.",
        "total_reward = progress + success_bonus + timeout_penalty."
      ]
    }
  }
}
```

## Logging Schema (Event Stream)
Emit one event per stage plus one per finalized trajectory step.

### Common envelope
```json
{
  "event_name": "rl.stage.completed",
  "ts": "2026-04-09T12:34:56Z",
  "run_id": "run_123",
  "trajectory_id": "traj_456",
  "episode_id": "ep_001",
  "step_idx": 5,
  "stage": "scoring",
  "status": "ok",
  "payload": {}
}
```

### Event types
- `rl.stage.completed`
  - payload = one `StageIntrospection` object.
- `rl.step.recorded`
  - payload = full `LineWorldStepRecord` including full `introspection`.
- `rl.step.failed`
  - payload includes `stage`, `error_type`, `error_message`, and available partial introspection.

## Step-by-Step Replay Design
A replay client should:
1. Load trajectory step records in ascending `step_idx`.
2. For each step, render three panes (generation/parsing/scoring).
3. Display `reasoning_trace` as ordered bullets with raw/intermediate values side-by-side.
4. Support a per-step **"Explain reward"** action that highlights `scoring.intermediate_output` + formula trace.

## Debug UI / JSON-first Contract
- JSON is the source of truth (`LineWorldStepRecord` + nested introspection).
- UI is optional and can be generated from JSON without domain-specific adapters.
- Minimal UI widgets:
  - timeline (step index, reward, done)
  - stage tabs
  - diff view for state changes (`position_before` → `position_after`)
  - reward equation card (`progress + success_bonus + timeout_penalty`)

## Acceptance Criteria
- Every recorded step includes all 3 stages.
- Each stage has non-empty `reasoning_trace`.
- Replay can reconstruct reward explanation without re-running environment logic.
