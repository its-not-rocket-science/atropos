# CLI Adapter Separation Note

## Summary

`BaseEnv` now depends on `atroposlib.cli.adapters.CliAdapter` instead of owning CLI-related implementation details in the environment runtime module. The legacy import path `atroposlib.envs.cli_adapter.CliAdapter` remains available as a shim so existing callers do not need immediate changes.

## Separation of concerns

- `atroposlib.envs.base.BaseEnv` focuses on orchestration concerns (`step`, `process`, `serve`, `evaluate`, logging, checkpointing).
- `atroposlib.cli.adapters.CliAdapter` owns CLI argument rendering and YAML/CLI config merge behavior.
- `atroposlib.envs.cli_adapter` is now a compatibility wrapper that re-exports the extracted adapter class.

## Why this helps

- Keeps environment classes lightweight and runtime-focused.
- Makes CLI config behavior easier to test in isolation without constructing full environments.
- Reduces coupling between runtime orchestration and command-line UX concerns while preserving current CLI behavior.
