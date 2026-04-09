# FastAPI runtime production-readiness audit (April 2026)

## Scope

Target runtime module: `src/atroposlib/api/server.py`.

## Risk matrix

| Risk area | Current risk | Why it matters | Research-safe | Internal-team-safe | Production-safe |
|---|---|---|---|---|---|
| In-memory state (`queue`, status map) | High | Process restart loses queue/status; multi-worker deployment diverges state by process | Acceptable for local experiments | Move status to shared store; keep queue in-memory only for low criticality | External durable queue + external status store required |
| Reset endpoint exposure | High | Accidental or malicious state wipe causes data loss and control-plane disruption | Allow only in isolated dev envs | Disable by default | Remove endpoint or gate behind strong admin auth + audit logging |
| Open CORS (`*`) | Medium-High | Browser clients from any origin can call public endpoints; increases token leakage and abuse surface | Accept if no auth and local notebook usage | Restrict origins to approved internal domains | Strict allowlist + TLS + origin review pipeline |
| Missing authentication | High | Anyone with network path can enqueue/reset/read status | Optional in sandbox | Require API token on writes | Strong authN/authZ (OIDC/JWT, scopes, service identities) |
| Queue durability | High | Jobs disappear on restart, crash, autoscaling, or rolling deploy | Accept in ephemeral research | At-least status durability, optional queue persistence | Durable broker with retries, dead-letter queues, idempotency keys |
| Concurrent mutation of `app.state` | Medium | Race conditions corrupt queue depth/status under concurrent requests | Serialize with a lock | Keep lock and typed state accessors | Replace with external atomic primitives/transactions |

## Hardening roadmap

### 1) Research-safe

- Keep in-memory queue and status for speed.
- Keep reset endpoint enabled for quick iteration.
- Allow broad CORS for local tools.
- Auth optional.
- Add typed state container and lock to avoid accidental races.

### 2) Internal-team-safe

- Disable reset endpoint in default runtime profile.
- Require API token for all mutating endpoints.
- Restrict CORS to known internal origins.
- Persist status to Redis/Postgres; keep in-memory queue only if workload tolerance is high.
- Add request logging with actor identity and endpoint.

### 3) Production-safe

- Remove in-memory queue as source of truth; move to durable broker (Redis Streams, RabbitMQ, SQS).
- Persist job state transitions in durable DB (Postgres preferred for auditability).
- Add strong authN/authZ (OIDC/JWT), role-based policy, and endpoint-level authorization.
- Remove reset endpoint from public surface; use an internal maintenance workflow with explicit approvals.
- Add idempotency keys, retry policies, DLQ, and replay tooling.
- Run with multiple workers and externalized shared state only.

## Example refactors implemented

1. Introduced typed runtime state with `AppRuntimeState` dataclass and helper accessor `get_runtime_state`.
2. Added explicit hardening tiers (`research-safe`, `internal-team-safe`, `production-safe`) with policy mapping.
3. Gated reset endpoint by policy rather than unconditional exposure.
4. Added write-endpoint auth dependency with `X-API-Token` verification for safer tiers.
5. Replaced ad hoc state mutation with lock-guarded mutation blocks.

## Persistence strategy recommendation (queue + status)

### Queue

- **Research-safe**: in-memory `deque`.
- **Internal-team-safe**: Redis list/stream acceptable when occasional data loss is tolerable.
- **Production-safe**: durable message broker with ack/retry semantics.

### Status

- Use a separate status store keyed by `job_id`.
- Start with Redis hash (fast, simple) or Postgres table (audit-friendly).
- Recommended production schema fields:
  - `job_id`, `state`, `created_at`, `updated_at`, `attempt`, `worker_id`, `last_error`.
- Persist every transition (`queued`, `running`, `succeeded`, `failed`) and include a monotonic version/timestamp for conflict resolution.

### Migration order

1. Keep API contract stable (`/jobs`, `/jobs/{id}`) while introducing storage adapters.
2. Add abstract queue/status interfaces and a memory-backed adapter.
3. Implement Redis/Postgres adapters and switch via configuration.
4. Run dual-write for one release window, then cut over reads.
5. Remove in-memory source-of-truth once confidence is high.
