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

Readiness reflects dependency health against the API server (`/health/ready`).

## Local deployment (Docker Compose)

### Start stack

```bash
docker compose -f docker-compose.deploy.yml up --build -d
```

### Verify health

```bash
curl -sf http://localhost:8000/health/live
curl -sf http://localhost:8000/health/ready
curl -sf http://localhost:9000/livez
curl -sf http://localhost:9000/readyz
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
curl -sf http://localhost:9000/readyz
```

## Smoke test instructions

Use these tests after each deployment:

1. **API liveness/readiness:** `/health/live`, `/health/ready`
2. **Worker liveness/readiness:** `/livez`, `/readyz`
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
