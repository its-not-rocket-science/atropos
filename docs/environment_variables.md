# Environment variable reference

This reference covers deployment-facing variables used by the API and worker artifacts.

## Runtime profile layering

The runtime now supports explicit configuration profiles selected by `ATROPOS_RUNTIME_PROFILE`:

| Profile | Purpose | Default tier | Default backend |
|---|---|---|---|
| `local-dev` | Developer workstation defaults for quick iteration | `research-safe` | `memory` |
| `ci` | CI pipelines with stricter policy but ephemeral storage | `internal-team-safe` | `memory` |
| `production` | Durable runtime deployment | `production-safe` | `redis` |

Profile defaults can be overridden, but `production` enforces durable + explicit config validation before startup.

### Dev vs production mode behavior

`ATROPOS_RUNTIME_PROFILE` now also drives a higher-level runtime mode:

- `local-dev` and `ci` run in `dev` mode (lightweight defaults for iteration and automation).
- `production` runs in explicit `production` mode (platform-safe defaults with fail-fast validation).

| Capability | Dev mode (`local-dev`, `ci`) | Production mode (`production`) |
|---|---|---|
| Store backend | Defaults to `memory`; durable backend optional | Must be durable (`ATROPOS_STORE_BACKEND=redis`) |
| Health endpoints | Available by default; requirement toggle defaults off | Must remain enabled (`ATROPOS_REQUIRE_HEALTH_ENDPOINTS=true`) |
| Log format | `json` by default, `pretty` allowed for local readability | Must resolve to structured JSON logs |
| Missing config handling | Optional env values allowed | Missing required settings fail startup immediately |
| Localhost defaults | Allowed for convenience | Rejected unless `ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS=true` |

Examples are available in:

- `examples/runtime-profiles/local-dev.env`
- `examples/runtime-profiles/ci.env`
- `examples/runtime-profiles/production.env.example`

## API server

| Variable | Required | Default | Description |
|---|---|---|---|
| `ATROPOS_RUNTIME_PROFILE` | No | `local-dev` | Runtime config layer. Values: `local-dev`, `ci`, `production`. |
| `ATROPOS_HARDENING_TIER` | No | Profile-specific | Runtime policy tier. Values: `research-safe`, `internal-team-safe`, `production-safe`. |
| `ATROPOS_STORE_BACKEND` | No | Profile-specific | Store backend selector (`memory` or `redis`). |
| `ATROPOS_REDIS_URL` | Yes for production | _(none)_ | Redis DSN for durable runtime backend. Production startup fails if unset. |
| `ATROPOS_API_TOKEN` | Required for auth-required tiers; always required in production profile | _(none)_ | API token validated from request header `X-API-Token`. |
| `ATROPOS_ALLOWED_ORIGINS` | Required in production profile | _(empty)_ | Comma-separated CORS origins for non-open tiers. Production startup fails if empty. |
| `ATROPOS_LOG_FORMAT` | No | library default | Optional runtime logger format override. |
| `ATROPOS_REQUIRE_HEALTH_ENDPOINTS` | Yes in production mode (default `true`) | `false` in dev mode | Enforces required health routes (`/health`, `/health/live`, `/health/ready`, `/health/dependencies`) at startup. |
| `ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS` | No | `false` | Allows localhost Redis URLs and localhost CORS origins in production mode when set to `true`. |

## Observability (runtime API)

| Variable | Required | Default | Description |
|---|---|---|---|
| `ATROPOS_TRACING_ENABLED` | No | `false` in `local-dev`/`ci`, `true` in `production` | Enables OpenTelemetry tracing setup. |
| `ATROPOS_TRACING_EXPORTER` | No | `otlp` | Exporter (`otlp`, `jaeger`, or `console`). |
| `ATROPOS_TRACING_ENDPOINT` | Required when tracing is enabled in production profile | exporter-specific fallback | Endpoint for trace export (e.g. OTLP collector URL). |
| `ATROPOS_TRACING_SERVICE_NAME` | No | `atropos-runtime` | OpenTelemetry service name. |
| `ATROPOS_TRACING_INSECURE` | No | `true` | Whether exporter transport is insecure (where applicable). |
| `ATROPOS_TRACING_SAMPLE_RATIO` | No | `1.0` | Trace sampling ratio in `[0.0, 1.0]`. |

### Production validation rules

When `ATROPOS_RUNTIME_PROFILE=production`, startup validation requires all of the following:

1. `ATROPOS_STORE_BACKEND=redis`
2. `ATROPOS_REDIS_URL` set
3. `ATROPOS_API_TOKEN` set
4. `ATROPOS_ALLOWED_ORIGINS` contains at least one origin
5. `ATROPOS_LOG_FORMAT` resolves to `json` (structured logging)
6. `ATROPOS_REQUIRE_HEALTH_ENDPOINTS` remains enabled
7. Localhost Redis/CORS defaults are rejected unless `ATROPOS_ALLOW_UNSAFE_LOCALHOST_DEFAULTS=true`
8. If `ATROPOS_TRACING_ENABLED=true`, `ATROPOS_TRACING_ENDPOINT` must be set

This prevents implicit localhost assumptions in production mode and makes durability/observability requirements explicit.

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

| Service | Liveness | Readiness | Dependencies |
|---|---|---|---|
| API | `/health/live` | `/health/ready` | `/health/dependencies` |
| Worker | `/livez` (or `/health/live`) | `/readyz` (or `/health/ready`) | `/depz` (or `/health/dependencies`) |
