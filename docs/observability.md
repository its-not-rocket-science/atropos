# Atropos Observability

Atropos runtime now exposes production-ready observability primitives:

- **Structured logging** via `atroposlib.observability.setup_json_logging()`.
- **Prometheus-compatible metrics** exposed by FastAPI at `GET /metrics`.
- **OpenTelemetry tracing hooks** around end-to-end runtime and ingestion spans.

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
