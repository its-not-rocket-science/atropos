# FastAPI Runtime Storage Migration Guide

## Why this change

The runtime server no longer relies on process-local `app.state` dictionaries/deques for queue and status metadata. It now uses a pluggable storage abstraction so you can run multiple API instances safely and survive process restarts.

## New architecture

- `RuntimeStore` protocol defines storage operations for enqueue, status fetch, and reset.
- `InMemoryStore` preserves existing behavior for local development and tests.
- `RedisStore` provides multi-instance coordination and fault-tolerant persistence.
- `PostgresStore` is included as a stub for future implementation.

## Breaking/API changes

- `build_runtime_app(...)` now accepts `store: RuntimeStore | None`.
- In production tier (`HardeningTier.PRODUCTION_SAFE`), default store is Redis and requires an explicit URL:
  - `ATROPOS_REDIS_URL` env var, or
  - `redis_url=...` passed to `build_runtime_app(...)`.
- Production tier now asserts `store.durable is True`; injecting `InMemoryStore` raises `ValueError`.
- `POST /jobs` supports optional `X-Idempotency-Key` and returns:
  - `job_id`
  - `queue_depth`
  - `deduplicated`
- `GET /health` now includes `store` backend name.

## Migration steps

1. Install runtime dependency for Redis deployments:
   ```bash
   pip install redis
   ```
2. Set `ATROPOS_REDIS_URL` for production pods:
   ```bash
   export ATROPOS_REDIS_URL="redis://redis.internal:6379/0"
   ```
3. Pass idempotency key from clients on retries:
   - `X-Idempotency-Key: <stable-request-key>`
4. (Optional) Explicitly inject store during app construction:
   ```python
   from atroposlib.api import build_runtime_app, HardeningTier, RedisStore

   app = build_runtime_app(
       tier=HardeningTier.PRODUCTION_SAFE,
       api_token="...",
       allowed_origins=["https://internal.example"],
       store=RedisStore.from_url("redis://redis.internal:6379/0"),
   )
   ```

## Operational notes

- Redis idempotency entries expire after 24 hours by default.
- Reset endpoint clears all keys under configured prefix.
- Use dedicated key prefixes per environment (`dev`, `staging`, `prod`) to isolate queues.

## Startup/shutdown lifecycle and probes

- Runtime startup now explicitly initializes the store backend before serving traffic.
- Graceful shutdown marks the API as not ready and closes backend resources.
- New probe endpoints:
  - `GET /health/live` for liveness (`200` unless startup fatally failed)
  - `GET /health/ready` for readiness (`200` only when initialized, dependencies healthy, and not shutting down)
  - `GET /health/dependencies` for direct dependency status (`redis`, `in_memory`, etc.)
- Readiness payload includes:
  - `store_durable` (whether backend is durable)
  - `recovered_items` (queue/scored state discovered at startup)

## Kubernetes deployment notes

Example probe wiring:

```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8011
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8011
  initialDelaySeconds: 5
  periodSeconds: 5

startupProbe:
  httpGet:
    path: /health/ready
    port: 8011
  failureThreshold: 30
  periodSeconds: 2
```

Recommended runtime settings:

- Set `terminationGracePeriodSeconds` high enough to finish in-flight requests.
- Use Redis (or another durable backend) in production so restart preserves critical request/idempotency state.
- Keep `ATROPOS_REDIS_URL` configured via Secret/ConfigMap.

## systemd deployment notes

Recommended service behaviors:

- Use `ExecStart` with your ASGI server command (for example `uvicorn ...`).
- Set `Restart=on-failure` to recover from crashes.
- Set `TimeoutStopSec` to allow graceful FastAPI shutdown hooks to run.
- Add `Environment=ATROPOS_REDIS_URL=...` for durable state.

Example snippet:

```ini
[Service]
ExecStart=/usr/bin/env uvicorn mymodule:app --host 0.0.0.0 --port 8011
Restart=on-failure
TimeoutStopSec=30
Environment=ATROPOS_REDIS_URL=redis://redis.internal:6379/0
```


## Restart recovery verification

For production-grade restart behavior, run with a shared durable store backend (Redis).

Expected outcomes across API process restarts:

- Job idempotency keys still deduplicate (`POST /jobs` returns same `job_id`).
- Existing `job_id` status remains queryable (`GET /jobs/{job_id}`).
- Scored ingestion request IDs still deduplicate retries (`POST /scored_data` / `/scored_data_list`).

Reference tests:

- `test_production_restart_recovers_job_and_dedup_state`
- `test_production_restart_recovers_scored_request_dedupe_state`
- `test_production_tier_rejects_inmemory_store`

See also: `docs/runtime_app_state_inventory_2026_04.md` for full `app.state` audit and classification.
