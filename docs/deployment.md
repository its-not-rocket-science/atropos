# Deployment guide

This guide documents deployable runtime services that exist in the current repository and maps them to concrete deployment artifacts.

## What is deployable today

The current runtime architecture is a 3-service deployment:

1. **Atropos Runtime API** (`atroposlib.api.asgi:app`) for jobs, scored data ingestion, environment registration, and health/metrics endpoints.
2. **Durable backing store** (Redis) required whenever `ATROPOS_RUNTIME_PROFILE=production`.
3. **Runtime worker service** (`python -m atroposlib.workers.runtime`) that exposes probe endpoints and continuously gates readiness on API readiness.

Artifacts in this repo that deploy those real code paths:

- API image: `docker/Dockerfile.api`
- Worker image: `docker/Dockerfile.worker`
- Local multi-service stack: `docker-compose.deploy.yml`
- Kubernetes manifests: `deploy/k8s/*.yaml`
- Runtime env examples: `examples/runtime-profiles/*.env`

## Health endpoint contract

### API (`:8000`)

- `GET /health`
- `GET /health/live`
- `GET /health/ready`
- `GET /health/dependencies`
- `GET /metrics`

### Worker (`:9000`)

- `GET /livez` and `GET /health/live`
- `GET /readyz` and `GET /health/ready`
- `GET /depz` and `GET /health/dependencies`
- `GET /metrics`

## Local deployment (Docker Compose)

### Start stack

```bash
docker compose -f docker-compose.deploy.yml up --build -d
```

### Smoke checks

```bash
curl -sf http://localhost:8000/health/live
curl -sf http://localhost:8000/health/ready
curl -sf http://localhost:8000/health/dependencies
curl -sf http://localhost:9000/livez
curl -sf http://localhost:9000/readyz
curl -sf http://localhost:9000/depz
```

### Auth + write-path check

```bash
curl -i -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -H 'X-API-Token: local-dev-token' \
  -d '{"task":"smoke"}'
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

### Verify rollout

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

## Production configuration notes

- Set `ATROPOS_RUNTIME_PROFILE=production`.
- Keep `ATROPOS_STORE_BACKEND=redis` and configure `ATROPOS_REDIS_URL`.
- Set `ATROPOS_API_TOKEN` via secret management.
- Provide non-empty `ATROPOS_ALLOWED_ORIGINS`.
- Keep `ATROPOS_REQUIRE_HEALTH_ENDPOINTS=true`.
- Use structured logs (`ATROPOS_LOG_FORMAT=json`).

For full variable descriptions and validation behavior, see `docs/environment_variables.md`.
