# Atropos Observability

Atropos runtime now exposes production-ready observability primitives:

- **Structured logging** via `atroposlib.observability.setup_json_logging()`.
- **Prometheus-compatible metrics** exposed by FastAPI at `GET /metrics`.
- **OpenTelemetry tracing hooks** around end-to-end runtime and ingestion spans.

## Audit scope: experiment metrics vs platform health

Atropos uses **two telemetry planes** that should not be conflated:

1. **Experiment metrics (W&B-compatible, model quality/business outcomes)**  
   Examples: reward curves, quality deltas, throughput deltas across control/treatment variants.
   These are used to decide *whether a model change is better*.
2. **Platform/service health metrics (Prometheus + logs + traces)**  
   Examples: API error rate, queue depth, duplicate ingestion counters, dependency readiness failures.
   These are used to decide *whether the control plane/runtime is healthy*.

Success criterion: operators can debug degraded control-plane behavior from runtime logs/metrics/traces
without opening experiment dashboards first.

## Instrumentation plan

### Control plane (FastAPI runtime API)

- Keep structured JSON logs with stable correlation keys (`env_id`, `request_id`, `batch_id`, `endpoint`).
- Record request middleware metrics for throughput, latency, status, and environment-level request volume.
- Emit queue/duplicate/state gauges and counters during scored-data ingestion.
- Capture error-path request logs (`request_failed`) and error counters even on unhandled exceptions.
- Expose metrics on `GET /metrics`.

### Runtime worker / orchestration loop

- Emit step start/finish structured logs with env, request, batch, worker, and rollout latency.
- Publish worker scaling/health gauges (worker count, utilization, trainer queue pressure, rate-limit ratio).
- Add worker dependency health metrics (`checks_total`, `failures_total`, `ready` gauge) and `/metrics`.
- Keep `failed_sends_total` and `eval_duration_seconds` for failure and slowness diagnosis.

### Optional tracing hooks

- Continue no-op-safe tracing wrapper so tracing can be disabled with zero code changes.
- Add request-level API span (`runtime.api.request`) plus worker dependency check span
  (`runtime_worker.dependency_check`) to connect ingress latency and control-plane readiness.

## Tracked signals

- `atropos_runtime_queue_depth{env=...}` (gauge)
- `atropos_buffered_groups{env=...}` (gauge)
- `atropos_runtime_groups_by_state{env=...,state=...}` (gauge)
- `atropos_ingestion_records_total{env=...}` (counter; use `rate()` for ingestion rate)
- `atropos_duplicate_ingestion_rejections_total{env=...,endpoint=...}` (counter)
- `atropos_batch_formation_latency_seconds{env=...}` (histogram)
- `atropos_worker_count{env=...}` (gauge)
- `atropos_rollout_latency_seconds{env=...}` (histogram)
- `atropos_failed_sends_total{env=...}` (counter)
- `atropos_eval_duration_seconds{env=...}` (histogram)
- `atropos_worker_utilization_ratio{env=...}` (gauge)
- `atropos_api_errors_total{path=...,status=...}` (counter)
- `atropos_api_requests_by_env_total{env=...,path=...,status=...}` (counter)
- `atropos_worker_dependency_checks_total{dependency=...}` (counter)
- `atropos_worker_dependency_failures_total{dependency=...}` (counter)
- `atropos_worker_dependency_ready{dependency=...}` (gauge)

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

## OpenTelemetry tracing

Tracing is configurable via environment variables and is disabled by default.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ATROPOS_TRACING_ENABLED` | `false` | Master toggle for tracing setup. |
| `ATROPOS_TRACING_SERVICE_NAME` | `atropos-runtime` | OpenTelemetry `service.name` resource attribute. |
| `ATROPOS_TRACING_EXPORTER` | `otlp` | Exporter: `otlp`, `jaeger`, or `console`. |
| `ATROPOS_TRACING_ENDPOINT` | unset | Export endpoint URL (exporter-specific). |
| `ATROPOS_TRACING_SAMPLE_RATIO` | `1.0` | Fraction of traces to sample (`0.0`–`1.0`). |

When enabled, tracing is initialized during runtime app startup (`build_runtime_app`) and spans
are emitted for these critical flows:

- `baseenv.environment_item_fetch`
- `baseenv.trajectory_collection`
- `baseenv.postprocess`
- `baseenv.send_to_api`
- `runtime.ingest_scored_data` and `runtime.ingest_scored_data.store`
- `runtime.batch_construction`
- `runtime.trainer_batch_fetch`

Each span includes attributes such as environment ID, request ID, worker count, group ID, and
record counts for debugging latency and failure propagation across distributed runs.

### Export traces to Jaeger

Jaeger (collector endpoint):

```bash
export ATROPOS_TRACING_ENABLED=true
export ATROPOS_TRACING_EXPORTER=jaeger
export ATROPOS_TRACING_ENDPOINT=http://localhost:14268/api/traces
```

You need the Jaeger exporter package installed:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-jaeger-thrift
```

### Export traces to Tempo (via OTLP)

Tempo typically ingests OTLP traces (HTTP/protobuf):

```bash
export ATROPOS_TRACING_ENABLED=true
export ATROPOS_TRACING_EXPORTER=otlp
export ATROPOS_TRACING_ENDPOINT=http://localhost:4318/v1/traces
```

Install OTLP exporter dependencies:

```bash
pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

## Grafana examples

Import one of:

- `docs/grafana/runtime-observability-dashboard.json`
- `docs/grafana/rl-rollout-dashboard.json`
- `docs/grafana/runtime-service-monitoring-dashboard.json`

Both dashboards expect a Prometheus datasource scraping the runtime `/metrics` endpoint.

## Operational questions and the primary signals

- **What is stuck?**  
  `atropos_runtime_queue_oldest_age_seconds`, `atropos_runtime_groups_by_state{state!="acknowledged"}`.
- **What is duplicated?**  
  `rate(atropos_duplicate_ingestion_rejections_total[5m])`,
  `rate(atropos_duplicate_ingestion_groups_total[5m])`.
- **What is slow?**  
  `histogram_quantile(0.95, rate(atropos_api_request_latency_seconds_bucket[5m]))`,
  `histogram_quantile(0.95, rate(atropos_rollout_latency_seconds_bucket[5m]))`.
- **What is failing?**  
  `rate(atropos_api_errors_total[5m])`, `rate(atropos_failed_sends_total[5m])`,
  `rate(atropos_worker_dependency_failures_total[5m])`, and `request_failed` logs.
- **What queue is growing?**  
  `deriv(atropos_runtime_queue_depth[10m]) > 0` and queue oldest-age trend.
- **Which env is noisy?**  
  `sum(rate(atropos_api_requests_by_env_total[5m])) by (env)` and per-env error ratio.

## Runbook: How to debug a degraded Atropos deployment

1. **Check dependency and process health**
   - API: `/health`, `/health/live`, `/health/ready`, `/health/dependencies`
   - Worker: `/livez`, `/readyz`
2. **Confirm scrape path and data freshness**
   - Verify Prometheus can scrape `/metrics` from API and worker services.
3. **Triage by question**
   - Stuck: look for high queue oldest age + growing non-acknowledged group counts.
   - Duplicated: inspect duplicate counters and request IDs in JSON logs.
   - Slow: inspect p95 API and rollout latency panels, then correlate with traces.
   - Failing: inspect API error-rate panel, failed send counters, worker dependency failures.
   - Noisy env: rank `api_requests_by_env_total` and compare error ratio by env.
4. **Use structured logs for fast narrowing**
   - Filter on `request_id`, then pivot by `env_id`, `endpoint`, `batch_id`.
   - Track `request_completed`/`request_failed` and `runtime_step_started`/`runtime_step_finished`.
5. **Use traces for cross-component timing**
   - Inspect `runtime.api.request` root spans.
   - Follow child spans (`runtime.ingest_scored_data*`, `baseenv.send_to_api`,
     `runtime_worker.dependency_check`) for bottlenecks and failure points.
