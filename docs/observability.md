# Atropos Observability

Atropos runtime now exposes production-ready observability primitives:

- **Structured logging** via `atroposlib.observability.setup_json_logging()`.
- **Prometheus-compatible metrics** exposed by FastAPI at `GET /metrics`.
- **OpenTelemetry tracing hooks** around `BaseEnv.step()` spans.

## Tracked signals

- `atropos_rollout_latency_seconds` (histogram)
- `atropos_runtime_queue_depth{env=...}` (gauge)
- `atropos_worker_utilization_ratio{env=...}` (gauge)
- `atropos_api_errors_total{path=...,status=...}` (counter)

## Runtime middleware

`build_runtime_app()` now installs API metrics middleware that records request throughput,
latency, and error counts for every route.

## BaseEnv tracing hooks

`BaseEnv.step()` now emits tracing spans (`baseenv.step`) and exports rollout + worker
utilization metrics with environment labels derived from `payload["env"]`.

## Grafana examples

Import one of:

- `docs/grafana/runtime-observability-dashboard.json`
- `docs/grafana/rl-rollout-dashboard.json`

Both dashboards expect a Prometheus datasource scraping the runtime `/metrics` endpoint.
