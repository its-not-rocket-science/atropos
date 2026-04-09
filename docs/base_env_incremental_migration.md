# BaseEnv Incremental Componentization Plan

## Responsibilities identified in the original monolith

`BaseEnv` currently combines:

1. Environment lifecycle contract (`reset`, `step`, termination semantics).
2. Worker orchestration/runtime policy.
3. API/model transport I/O.
4. Logging/telemetry.
5. Checkpointing/recovery.
6. CLI argument generation.
7. YAML + CLI config merge behavior.

## New components introduced

- `EnvRuntime`: worker orchestration.
- `EnvTransportClient`: API transport boundary.
- `EnvLogger`: structured event logging.
- `EnvCheckpointManager`: checkpoint persistence.
- `EnvCliBuilder`: CLI argument construction.
- `EnvConfigMerger`: YAML/CLI merge strategy.
- `BaseEnv`: compatibility facade that delegates to the components above.

## New file layout

```text
src/atroposlib/
  envs/
    __init__.py
    base.py
    components.py
```

## Minimal class skeletons

See:

- `src/atroposlib/envs/base.py`
- `src/atroposlib/envs/components.py`

These provide minimal implementations intended for incremental adoption while
keeping the previous `BaseEnv` surface area available through compatibility
methods (`step`, `reset`, `build_cli_args`, `merge_yaml_and_cli`, and legacy
aliases like `call_api`/`checkpoint`).

## Incremental migration sequence (no greenfield rewrite)

### Phase 1: Internal delegation only
- Keep `BaseEnv` import path and method names unchanged.
- Delegate old in-class logic to component collaborators.

### Phase 2: Per-responsibility adapter hardening
- Replace default components with production adapters one-by-one
  (`EnvTransportClient`, `EnvRuntime`, etc.).
- Preserve output shapes and CLI flags exactly.

### Phase 3: Controlled extensibility
- Allow dependency injection for each collaborator.
- Add tests per component boundary and facade parity tests.

### Phase 4: Deprecation runway
- Document legacy alias methods.
- Warn before eventual removal only after users migrate to component APIs.

## CLI behavior preservation notes

- `EnvCliBuilder.build()` emits deterministic sorted flags for stable output.
- `EnvConfigMerger.merge()` keeps current precedence: CLI overrides YAML.

