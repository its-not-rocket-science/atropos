# Atropos Observability

Atropos runtime now exposes production-ready observability primitives:

- **Structured logging** via `atroposlib.observability.setup_json_logging()`.
- **Prometheus-compatible metrics** exposed by FastAPI at `GET /metrics`.
- **OpenTelemetry tracing hooks** around `BaseEnv.step()` spans.

## Tracked signals

- `atropos_runtime_queue_depth{env=...}` (gauge)
- `atropos_buffered_groups{env=...}` (gauge)
- `atropos_ingestion_records_total{env=...}` (counter; use `rate()` for ingestion rate)
- `atropos_duplicate_ingestion_rejections_total{env=...,endpoint=...}` (counter)
- `atropos_batch_formation_latency_seconds{env=...}` (histogram)
- `atropos_worker_count{env=...}` (gauge)
- `atropos_rollout_latency_seconds{env=...}` (histogram)
- `atropos_failed_sends_total{env=...}` (counter)
- `atropos_eval_duration_seconds{env=...}` (histogram)
- `atropos_worker_utilization_ratio{env=...}` (gauge)
- `atropos_api_errors_total{path=...,status=...}` (counter)

## Runtime middleware

`build_runtime_app()` now installs API metrics middleware that records request throughput,
latency, and error counts for every route.

## Prometheus scraping

Example `prometheus.yml` scrape config:

```yaml
scrape_configs:
  - job_name: "atropos-runtime"
    metrics_path: /metrics
    static_configs:
      - targets:
          - "localhost:8000"
```

For ingestion throughput in Grafana/Prometheus:

```promql
sum(rate(atropos_ingestion_records_total[5m])) by (env)
```

## BaseEnv tracing hooks

`BaseEnv.step()` now emits tracing spans (`baseenv.step`) and exports rollout + worker
utilization metrics with environment labels derived from `payload["env"]`.

## Grafana examples

Import one of:

- `docs/grafana/runtime-observability-dashboard.json`
- `docs/grafana/rl-rollout-dashboard.json`
- `docs/grafana/runtime-service-monitoring-dashboard.json`

Both dashboards expect a Prometheus datasource scraping the runtime `/metrics` endpoint.
