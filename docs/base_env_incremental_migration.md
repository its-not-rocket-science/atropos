# BaseEnv Component Migration Guide

`BaseEnv` has been refactored into composable collaborators to improve testability,
dependency injection, and control over side effects.

## New module structure

```text
src/atroposlib/envs/
  base.py                # thin orchestration facade
  worker_manager.py      # async-worker policy, backlog, scaling decisions
  transport_client.py    # API communication + retry policy
  logging_manager.py     # event logging + metric recording hooks (e.g., W&B adapters)
  checkpoint_manager.py  # checkpoint persistence API
  env_logic.py           # user-defined environment business logic contract
  components.py          # backward-compatibility aliases for old imports
```

## Responsibility split

1. `WorkerManager`
   - owns queue/backlog state
   - computes worker scaling decisions
   - orchestrates step runtime payloads
2. `TransportClient`
   - owns request boundary
   - applies retry behavior
3. `LoggingManager`
   - captures structured events
   - records metrics in a dedicated channel
4. `CheckpointManager`
   - saves snapshots
   - exposes recovery helpers (`latest`)
5. `EnvLogic`
   - defines user logic only (`prepare_step`, `finalize_step`)

## Refactored `BaseEnv`

`BaseEnv` is now a thin orchestrator:

1. `env_logic.prepare_step(payload)`
2. `worker_manager.orchestrate(...)`
3. `transport_client.send(...)`
4. `env_logic.finalize_step(...)`
5. `checkpoint_manager.save(...)`
6. `logging_manager.log_event(...)`

This ordering is explicit and easy to test in isolation.

## Migration examples

### Before

```python
from atroposlib.envs.base import BaseEnv

env = BaseEnv()
result = env.step({"task": "x"}, worker_count=2)
```

### After (no changes required)

```python
from atroposlib.envs.base import BaseEnv

env = BaseEnv()
result = env.step({"task": "x"}, worker_count=2)
```

### After (recommended dependency injection)

```python
from atroposlib.envs.base import BaseEnv
from atroposlib.envs.worker_manager import WorkerManager
from atroposlib.envs.transport_client import TransportClient
from atroposlib.envs.logging_manager import LoggingManager
from atroposlib.envs.checkpoint_manager import CheckpointManager
from atroposlib.envs.env_logic import PassthroughEnvLogic

env = BaseEnv(
    worker_manager=WorkerManager(max_workers=64),
    transport_client=TransportClient(max_retries=3),
    logging_manager=LoggingManager(),
    checkpoint_manager=CheckpointManager(),
    env_logic=PassthroughEnvLogic(),
)
```

## Backward compatibility shim

Legacy import paths continue to work via `components.py` aliases:

- `EnvRuntime -> WorkerManager`
- `EnvTransportClient -> TransportClient`
- `EnvLogger -> LoggingManager`
- `EnvCheckpointManager -> CheckpointManager`

Legacy `BaseEnv` aliases (`orchestrate_workers`, `call_api`, `log_event`,
`checkpoint`) are still available for incremental adoption.
