# Control-plane integration test package

This package extends runtime inference-tracking checks into end-to-end control-plane coverage across API, data-plane ingestion, and runtime durability behavior.

## Coverage in this package

`test_control_plane_runtime_integration.py` validates:

- environment registration and listing (`POST/GET /environments`)
- queue + buffer lifecycle progression for scored data groups
- batch formation and bounded trainer fetch (`GET /scored_data_list?limit=...`)
- idempotency + dedupe for jobs and scored-data ingestion
- runtime status reporting (`/jobs/{id}`, `/health/ready`)
- restart recovery for durable backends (RedisStore with persisted fake Redis state)
- production-mode safety (`HardeningTier.PRODUCTION_SAFE`) with durable-store enforcement and API token checks

## Fixtures

Shared fixtures live in `conftest.py`:

- `runtime_backend`: parametrized local/durable mode fixture
  - `local`: `InMemoryStore`
  - `durable`: `RedisStore` wired to a CI-safe `FakeRedis`
- `runtime_client`: in-process FastAPI client bound to the selected backend
- `durable_backend`: explicit durable fixture for restart recovery tests

## Run locally

Run just this package:

```bash
pytest tests/control_plane_integration -q
```

Run the control-plane integration file only:

```bash
pytest tests/control_plane_integration/test_control_plane_runtime_integration.py -q
```

Run with verbose backend parametrization output:

```bash
pytest tests/control_plane_integration/test_control_plane_runtime_integration.py -vv
```

## CI-friendliness

- No external services are required.
- Durable-mode paths use an in-memory Redis double.
- Tests are deterministic and avoid stress/load timing assumptions.

## Remaining blind spots

- No true multi-process/multi-node contention validation against a real Redis server.
- No chaos/fault-injection around partial writes, network partitions, or dependency brownouts.
- No long-lived TTL expiry/idempotency window rollover checks.
- No production ingress/CORS/proxy behavior validation outside in-process TestClient.
