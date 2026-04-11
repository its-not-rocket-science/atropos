# Canonical Trajectory Schema (Atropos v1.0.0)

This specification defines a **versioned canonical trajectory record** for Atropos as a reusable data layer for:

- RLHF pipelines
- Offline RL datasets
- Native Atropos RL environments

## Schema envelope

- `schema_name` (string, const): `atropos.trajectory`
- `schema_version` (string, const): `1.0.0`
- `trajectory_id` (string)
- `episode_id` (string)
- `created_at` (ISO-8601 UTC timestamp)
- `steps` (array of `TrajectoryStep`)
- `metadata` (object)
- `environment_state` (optional JSON value)

## Step schema

Each `TrajectoryStep` contains:

- `step_idx` (int, contiguous starting at 1)
- `tokens_in` (list[int])
- `tokens_out` (list[int])
- `reward`:
  - `total` (float)
  - `components` (map[str, float])
  - `source` (string)
- `action` (JSON value, optional)
- `observation` (JSON value, optional)
- `next_observation` (JSON value, optional)
- `done` (bool)
- `metadata` (object)

## Serialization formats

### JSON

Canonical JSON representation uses one envelope per trajectory and preserves nested objects.

### Arrow-compatible rows

Canonical row flattening uses one row per step with trajectory-level fields repeated:

- `schema_name`, `schema_version`, `trajectory_id`, `episode_id`, `created_at`
- `step_idx`, `tokens_in`, `tokens_out`
- `reward_total`, `reward_components`, `reward_source`
- `action`, `observation`, `next_observation` (JSON-encoded strings)
- `done`, `step_metadata`, `trajectory_metadata`, `environment_state`

## Conversion tools

Atropos now provides converters for:

- `LineWorldEnv` in-memory step history
- `LineWorldEnv.save_rollout` JSON payloads
- RLHF preference/pair records (`prompt`, `response`, token fields, reward/score fields)
- Offline RL transition datasets (`observation`, `action`, `next_observation`, `reward`, `done`)

## API surface

Implemented under `atropos.trajectory`:

- Schema: `TrajectoryRecord`, `TrajectoryStep`, `RewardSignal`, `build_schema_spec`
- Serde: `to_json`, `from_json`, `write_json`, `read_json`, `to_arrow_rows`, `from_arrow_rows`
- Conversion: `from_line_world_history`, `from_line_world_rollout`, `from_rlhf_pairs`, `from_offline_rl_transitions`
