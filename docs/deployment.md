# Deployment guide

This guide provides production deployment artifacts for Atropos runtime services:

- API server container (`docker/Dockerfile.api`)
- Runtime worker container (`docker/Dockerfile.worker`)
- Local multi-service compose stack (`docker-compose.deploy.yml`)
- Kubernetes manifests (`deploy/k8s/*.yaml`)

## Services

### API server

The API container runs:

```bash
uvicorn atroposlib.api.asgi:app --host 0.0.0.0 --port 8000
```

It supports hardening tier + Redis-backed durability through environment variables.

### Runtime worker

The worker container runs:

```bash
python -m atroposlib.workers.runtime
```

This worker exposes:

- `GET /livez` for liveness
- `GET /readyz` for readiness
- `GET /depz` for dependency health
- aliases under `/health/*` (`/health/live`, `/health/ready`, `/health/dependencies`)

Readiness reflects dependency health against the API server (`/health/ready`).

## Health endpoint contract

### API (`:8000`)

| Endpoint | Purpose | Healthy | Degraded | Unavailable |
|---|---|---|---|---|
| `/health/live` | Process liveness | `200` + `{"status":"alive","health_state":"healthy"}` | n/a | `503` + `{"status":"error","health_state":"unavailable",...}` |
| `/health/ready` | Traffic readiness gate | `200` + `{"status":"ready","health_state":"healthy","control_plane_ready":true,"backing_store_healthy":true,...}` | `503` + `{"status":"not_ready","health_state":"degraded","control_plane_ready":true,"backing_store_healthy":false,...}` | `503` + `{"status":"not_ready","health_state":"unavailable",...}` |
| `/health/dependencies` | Backing store dependency state | `200` + `{"status":"ok","health_state":"healthy","dependency":"redis|in_memory",...}` | `503` + `{"status":"degraded","health_state":"degraded",...}` | n/a |

### Worker (`:9000`)

| Endpoint | Purpose | Healthy | Degraded | Unavailable |
|---|---|---|---|---|
| `/livez` (or `/health/live`) | Worker process liveness | `200` + `{"status":"alive","health_state":"healthy",...}` | n/a | `503` + `{"status":"starting|terminating","health_state":"unavailable",...}` |
| `/readyz` (or `/health/ready`) | Worker readiness gate | `200` + `{"status":"ready","health_state":"healthy","control_plane_ready":true,"dependency_healthy":true,...}` | `503` + `{"status":"not_ready","health_state":"degraded",...}` | `503` + `{"status":"not_ready","health_state":"unavailable",...}` |
| `/depz` (or `/health/dependencies`) | Worker control-plane dependency state | `200` + `{"status":"ok","health_state":"healthy","dependency":"runtime_api",...}` | `503` + `{"status":"degraded","health_state":"degraded",...}` | `503` + `{"status":"degraded","health_state":"unavailable",...}` |

## Local deployment (Docker Compose)

### Start stack

```bash
docker compose -f docker-compose.deploy.yml up --build -d
```

### Verify health

```bash
curl -sf http://localhost:8000/health/live
curl -sf http://localhost:8000/health/ready
curl -sf http://localhost:8000/health/dependencies
curl -sf http://localhost:9000/livez
curl -sf http://localhost:9000/readyz
curl -sf http://localhost:9000/depz
```

### Stop stack

```bash
docker compose -f docker-compose.deploy.yml down
```

## Kubernetes deployment

### Apply manifests

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/secret.yaml
kubectl apply -f deploy/k8s/redis.yaml
kubectl apply -f deploy/k8s/api.yaml
kubectl apply -f deploy/k8s/worker.yaml
```

### Check rollout

```bash
kubectl -n atropos rollout status deploy/atropos-redis
kubectl -n atropos rollout status deploy/atropos-api
kubectl -n atropos rollout status deploy/atropos-worker
```

### Port-forward smoke checks

```bash
kubectl -n atropos port-forward svc/atropos-api 8000:8000
kubectl -n atropos port-forward deploy/atropos-worker 9000:9000
curl -sf http://localhost:8000/health/ready
curl -sf http://localhost:8000/health/dependencies
curl -sf http://localhost:9000/readyz
curl -sf http://localhost:9000/depz
```

## Smoke test instructions

Use these tests after each deployment:

1. **API liveness/readiness/dependencies:** `/health/live`, `/health/ready`, `/health/dependencies`
2. **Worker liveness/readiness/dependencies:** `/livez`, `/readyz`, `/depz`
3. **API auth enforcement:** call write endpoint without `X-API-Token` and expect `401`
4. **Redis-backed persistence path:** enqueue a job, restart API pod, verify readiness recovers

Example write-path check:

```bash
curl -i -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -H 'X-API-Token: local-dev-token' \
  -d '{"task":"smoke"}'
```

## Environment variable reference

See `docs/environment_variables.md` for the full runtime deployment environment contract.

## Readiness failure examples

1. **Backing store outage (API degraded):**
   - Trigger: Redis unavailable/network partition.
   - Expected:
     - `/health/live` returns `200`.
     - `/health/dependencies` returns `503` with `health_state=degraded`.
     - `/health/ready` returns `503` with `control_plane_ready=true`, `backing_store_healthy=false`, `health_state=degraded`.

2. **Startup failure (API unavailable):**
   - Trigger: invalid production configuration or startup exception.
   - Expected:
     - `/health/live` returns `503` with `health_state=unavailable`.
     - `/health/ready` returns `503` with `health_state=unavailable`.

3. **Control-plane unavailable (worker degraded/unavailable):**
   - Trigger: API `/health/ready` fails or API DNS endpoint unreachable.
   - Expected:
     - `/livez` usually remains `200` once started.
     - `/depz` returns `503` with `status=degraded`.
     - `/readyz` returns `503` with `dependency_healthy=false`.

## systemd and reverse-proxy probe mapping

### systemd (worker example)

Use `ExecStartPre` probe to block startup until API readiness:

```ini
[Service]
ExecStartPre=/usr/bin/curl -fsS http://127.0.0.1:8000/health/ready
ExecStart=/usr/bin/python -m atroposlib.workers.runtime
Restart=always
```

### Reverse proxies

Route external health paths directly to internal probes without auth mutation:

- API upstream: `/health/live`, `/health/ready`, `/health/dependencies`
- Worker upstream: `/livez`, `/readyz`, `/depz` (or `/health/*` aliases)

For NGINX, keep short upstream timeouts (1-2s) for probe locations so orchestrators quickly detect degraded dependencies.
