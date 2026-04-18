# Runtime `app.state` Inventory and Control-Plane Durability Audit (2026-04)

This document audits mutable FastAPI `app.state` usage in `src/atroposlib/api/server.py` and classifies each field by operational durability requirements.

## Scope

- API module audited: `src/atroposlib/api/server.py`
- Control-plane backing store: `AtroposStore` protocol with `RedisStore` (durable) and `InMemoryStore` (dev/test only)

## Mutable `app.state` fields currently used

### 1) `runtime_store`

- **Type:** `AtroposStore`
- **Purpose:** Request-time dependency access and compatibility accessor via `get_runtime_state(...)`.
- **Classification:** **safe to remain ephemeral**.
- **Rationale:** This is not control-plane data itself; it is a process-local pointer to the backend that owns durable state.

## Previously process-local fields moved off `app.state`

The following mutable lifecycle fields were intentionally moved to an internal `RuntimeProcessState` instance (closure-local in app factory), because they are process lifecycle metadata and not queue/control-plane state:

- `is_shutting_down`
- `ready`
- `startup_error`
- `store_startup_state`

### Classification

- **safe to remain ephemeral** (process lifecycle only).
- **not a source of truth for queue/buffer/batch/registration/status/dedupe data**.

## Control-plane state durability mapping

Critical operational state is persisted behind `AtroposStore`:

- Job queue and job status (`enqueue_job`, `get_job_status`)
- Idempotency dedupe for jobs
- Scored-data ingestion request dedupe
- Group-level acceptance/buffering/batching/delivery/ack status
- Scored-data environment record lists
- Queue depth/age metrics source data

### Store requirements by deployment mode

- **Production (`HardeningTier.PRODUCTION_SAFE`)**
  - Must use a **durable** backend (`store.durable == True`), enforced by assertion at app build time.
  - Default backend is Redis, requiring explicit Redis URL when no store is injected.
- **Research/Internal tiers**
  - May use `InMemoryStore` for local/dev workflows.

## Derived/cache-only state

The following are derived view/state and may remain process-local:

- Startup readiness envelope (`ready`, `startup_error`, `is_shutting_down`)
- Startup snapshot (`store_startup_state`: recovered item count + dependency health at startup)
- Prometheus in-process aggregations (observability counters/gauges), which are telemetry and not control-plane source of truth.

## Production safety assertions

Production app construction now fails fast if a non-durable store is configured:

- Error: `Production-safe tier requires a durable store backend; got backend='memory'`

This prevents accidental in-memory fallback in production runtime profiles.
