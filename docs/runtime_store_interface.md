# Runtime Store Interface (`AtroposStore`)

`AtroposStore` is the control-plane/storage abstraction for the FastAPI runtime server.

## Required capabilities

Every backend implementation must provide:

1. **Environment registration**
   - `register_environment(environment_id, now)`
   - `list_registered_environments()`
2. **Queueing + dedupe for jobs**
   - `enqueue_job(job_id, now, idempotency_key)`
   - `get_job_status(job_id)`
3. **Buffering + dedupe for scored data**
   - `ingest_scored_data(request_id, groups)`
   - group lifecycle transitions: `accepted -> buffered -> batched -> delivered -> acknowledged`
4. **Batch construction support**
   - `list_scored_data(environment_id, limit)` returns bounded record slices
5. **Status tracking + queue metrics**
   - `get_scored_group_status(environment_id, group_id)`
   - `get_scored_queue_metrics(now)`
6. **Operational lifecycle**
   - `startup()`, `shutdown()`, `dependency_health()`, `reset()`

## Built-in backends

- **`InMemoryStore`** (`durable=False`): process-local backend for local dev/tests.
- **`RedisStore`** (`durable=True`): production-grade durable backend for restart recovery and multi-instance deployment.
- **`PostgresStore`**: reserved stub for future implementation (not wired for production use).

## API-server integration notes

- Runtime API now exposes explicit environment control-plane routes:
  - `POST /environments`
  - `GET /environments`
- Runtime writes/reads also auto-register environment IDs to prevent bypasses where environment state was only implicit in payload handling.

## Compatibility

`RuntimeStore` remains as a backward-compatibility alias to `AtroposStore`.
