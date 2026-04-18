# Environment variable reference

This reference covers deployment-facing variables used by the API and worker artifacts.

## API server

| Variable | Required | Default | Description |
|---|---|---|---|
| `ATROPOS_HARDENING_TIER` | No | `production-safe` (in ASGI entrypoint) | Runtime policy tier. Values: `research-safe`, `internal-team-safe`, `production-safe`. |
| `ATROPOS_REDIS_URL` | Yes for durable production | `redis://localhost:6379/0` | Redis DSN used by production-safe tier storage backend. |
| `ATROPOS_API_TOKEN` | Required when tier requires auth | _(none)_ | API token validated from request header `X-API-Token`. |
| `ATROPOS_ALLOWED_ORIGINS` | No | _(empty)_ | Comma-separated CORS origins for non-open tiers. |
| `ATROPOS_LOG_FORMAT` | No | library default | Optional runtime logger format override. |

## Worker runtime

| Variable | Required | Default | Description |
|---|---|---|---|
| `ATROPOS_API_BASE_URL` | Yes | `http://atropos-api:8000` | API base URL that worker probes for readiness dependency checks. |
| `WORKER_HOST` | No | `0.0.0.0` | Host bind interface for worker health HTTP server. |
| `WORKER_PORT` | No | `9000` | Port bind for worker health HTTP server. |
| `WORKER_POLL_INTERVAL_SECONDS` | No | `5` | Seconds between worker dependency checks. |
| `WORKER_REQUEST_TIMEOUT_SECONDS` | No | `2` | HTTP timeout used for each worker dependency check. |
| `ATROPOS_LOG_LEVEL` | No | `INFO` | Logging level for worker process output. |

## Probe endpoints

| Service | Liveness | Readiness |
|---|---|---|
| API | `/health/live` | `/health/ready` |
| Worker | `/livez` | `/readyz` |
