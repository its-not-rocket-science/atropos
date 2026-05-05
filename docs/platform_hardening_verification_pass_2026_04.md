# Platform Hardening Verification Pass (2026-04-24)

Date: 2026-04-24

This pass re-verifies implementation status directly from current code and tests (without assuming prior refactors were correct), and closes any identified gaps.

## Summary status

| Area | Status | Notes |
|---|---|---|
| Runtime decoupling from `BaseEnv` | **Complete** | Runtime orchestration is delegated to protocol-driven collaborators and wiring factory. |
| Durable store-backed API control plane | **Complete** | FastAPI control plane supports `InMemoryStore` and durable `RedisStore` with production durability enforcement. |
| Idempotent ingestion | **Complete** | Request-level and group-level idempotency semantics implemented in both store backends. |
| Structured logging | **Complete** | API and runtime controller emit context-rich structured logs; production profile requires JSON logging. |
| Metrics / observability | **Complete** | Prometheus-style metrics, API middleware latency/error accounting, queue/ingestion gauges/counters, tracing spans. |
| Readiness / liveness semantics | **Complete** | Distinct `/health/live`, `/health/ready`, and dependency health endpoints with status-code semantics. |
| Restart recovery | **Complete** | Redis-backed control plane preserves idempotency and queued state across app restarts; covered in tests. |
| Integration and concurrency tests | **Complete** | E2E runtime tests and high-concurrency control-plane tests exist and validate dedupe + pressure paths. |
| Deployable artifacts | **Complete** | Docker API/worker images + compose + Kubernetes manifests include probes/config wiring. |
| Explicit prod mode | **Closed in this pass** | Added strict profile↔tier validation to enforce explicit production mode semantics. |

## Evidence by item

### 1) Runtime decoupling from `BaseEnv` — Complete

- `BaseEnv` is a thin facade; orchestration sits in `RuntimeController` and compatibility adapter layers. (`src/atroposlib/envs/base.py`)
- Protocol contracts define seams (`ItemSource`, `BacklogManager`, `SendToApiPath`, `RolloutCollector`, `EvalRunner`). (`src/atroposlib/envs/runtime_interfaces.py`)
- `RuntimeController` uses these interfaces end-to-end (prepare → orchestrate → transport → collect). (`src/atroposlib/envs/runtime_controller.py`)
- Decomposition tests verify wiring and compatibility behavior. (`tests/test_base_env_runtime_decomposition.py`)

### 2) Durable store-backed API control plane — Complete

- Store protocol and durable startup/dependency metadata are formalized in `AtroposStore`. (`src/atroposlib/api/storage.py`)
- `RedisStore` implements durable queue/status/ingestion/environment operations. (`src/atroposlib/api/storage.py`)
- Production-safe tier rejects non-durable store backends. (`src/atroposlib/api/server.py`)
- Tests assert production tier requires durable store and works with injected Redis backend. (`tests/test_runtime_server_storage.py`)

### 3) Idempotent ingestion — Complete

- Job enqueue idempotency via `X-Idempotency-Key` and backend-specific dedupe maps/keys. (`src/atroposlib/api/server.py`, `src/atroposlib/api/storage.py`)
- Scored data ingestion requires request identity (`X-Request-ID` or `X-Idempotency-Key`). (`src/atroposlib/api/server.py`)
- Request-level and group-level dedupe in both in-memory and Redis stores. (`src/atroposlib/api/storage.py`)
- E2E tests cover duplicate retries and partial failure retry paths. (`tests/test_api_runtime_e2e_integration.py`)

### 4) Structured logging — Complete

- API middleware and write endpoints emit structured context fields through `build_log_context`. (`src/atroposlib/api/server.py`, `src/atroposlib/logging_utils.py`)
- Runtime controller emits structured start/finish event logs. (`src/atroposlib/envs/runtime_controller.py`)
- Production config validation requires JSON logs. (`src/atroposlib/api/runtime_config.py`)
- Structured logging behavior is tested. (`tests/test_structured_logging.py`)

### 5) Metrics / observability — Complete

- Observability registry provides API, queue, ingestion, worker, and dependency metrics plus tracing controls. (`src/atroposlib/observability.py`)
- API middleware records request counts/latency/status and error paths. (`src/atroposlib/api/server.py`)
- Ingestion path records queue/group/duplicate/latency metrics. (`src/atroposlib/api/server.py`)
- `/metrics` endpoint exposed for scraping. (`src/atroposlib/api/server.py`)

### 6) Readiness / liveness semantics — Complete

- Distinct endpoints: `/health`, `/health/live`, `/health/ready`, `/health/dependencies`. (`src/atroposlib/api/server.py`)
- Readiness combines control-plane state, store startup/health, and shutdown flag; returns `503` when not ready. (`src/atroposlib/api/server.py`)
- ASGI bootstrap can assert required health routes are present. (`src/atroposlib/api/asgi.py`)

### 7) Restart recovery — Complete

- Redis store startup reports recovered queue/group counts and dependency status. (`src/atroposlib/api/storage.py`)
- Restart tests validate persistence of queue + idempotency state across app instances. (`tests/test_runtime_server_storage.py`)

### 8) Integration and concurrency tests — Complete

- Runtime E2E integration tests run against in-memory and Redis doubles. (`tests/test_api_runtime_e2e_integration.py`)
- Control-plane concurrency tests cover high parallel submissions, duplicate storms, mixed load, and polling during spikes. (`tests/control_plane_integration/test_control_plane_concurrency.py`)

### 9) Deployable artifacts — Complete

- API container artifact exists and runs ASGI runtime entrypoint. (`docker/Dockerfile.api`)
- Compose deployment wires Redis + API + worker with healthchecks and production profile defaults. (`docker-compose.deploy.yml`)
- Kubernetes manifests include readiness/liveness probes for API and worker. (`deploy/k8s/api.yaml`, `deploy/k8s/worker.yaml`)

### 10) Explicit prod mode — Closed in this pass

**Gap identified:** prior validation enforced production constraints only when profile was already `production`, but did not prevent selecting `production-safe` hardening tier from non-production profile.

**Fix implemented:**

- Enforce `production-safe` tier requires `ATROPOS_RUNTIME_PROFILE=production`.
- Enforce `ATROPOS_RUNTIME_PROFILE=production` requires `ATROPOS_HARDENING_TIER=production-safe`.

Implemented in `RuntimeDeploymentConfig.validate_for_runtime`. (`src/atroposlib/api/runtime_config.py`)

Added regression tests for both directions of the invariant. (`tests/test_runtime_config_profiles.py`)

## Remaining items

No unresolved platform-hardening gaps were found for the listed scope after this pass.
