# Runtime failure-recovery behavior (explicit contract)

This document defines explicit and testable recovery rules for Atropos API/runtime fault scenarios.

## Scope

Applies to:

- `src/atroposlib/api/server.py`
- `src/atroposlib/api/storage.py` (`InMemoryStore`, `RedisStore`)
- runtime ingestion endpoints: `POST /scored_data`, `POST /scored_data_list`
- runtime queue endpoint: `POST /jobs`
- environment registration endpoint: `POST /environments`

## Recovery rules

### 1) API restart

- **Durable backend (`RedisStore`)**: previously written queue state, idempotency keys, and scored-group lifecycle state are recovered on restart.
- **In-memory backend**: state is process-local and is not recovered after process restart.
- Readiness reflects recovered item count and dependency health (`/health/ready`).

### 2) Worker restart

- Workers must retry with the same `X-Request-ID`/idempotency key for a logical operation.
- API request-level deduplication guarantees that a restarted worker replay does not duplicate accepted records.

### 3) Store outage / dependency failure

- Readiness and dependency health endpoints report degraded status when backing dependency is unhealthy.
- Runtime write/read operations that hit store exceptions return **HTTP 503** with explicit operation context (`Runtime store unavailable during <endpoint>`), replacing implicit 500 behavior.

### 4) Duplicate environment registration

- `POST /environments` is idempotent.
- First registration returns `{created: true}`; subsequent duplicate registrations return `{created: false}`.
- For durable stores this remains true across API restarts.

### 5) Partial ingestion

- Request-level status can be `completed` or `partial_failed`.
- Failed groups do not block accepted groups.
- Retrying with the same request id accepts only previously unaccepted groups.

### 6) Batch delivery interruption

- Group lifecycle now includes explicit `interrupted` state.
- If delivery to the scored-data list fails mid-group:
  - group transitions to `interrupted`
  - request result is `partial_failed`
  - durable group-claim key is released so retry can succeed safely
- Retry with same request id can recover interrupted groups without duplicating already acknowledged groups.

## Group lifecycle state machine

`accepted -> buffered -> batched -> delivered -> acknowledged`

Failure branch:

`accepted|buffered|batched -> interrupted` (recoverable by retry)

## Observable outcomes

- `/health/ready` and `/health/dependencies` surface control-plane and dependency state.
- Ingestion response includes `status`, `failed_groups`, `duplicate_groups`, and `deduplicated`.
- Group status includes `interrupted_at` timestamp when interruption occurs.
