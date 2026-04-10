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
- In production tier (`HardeningTier.PRODUCTION_SAFE`), default store is Redis using:
  - `ATROPOS_REDIS_URL` env var, or
  - `redis://localhost:6379/0` fallback.
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
