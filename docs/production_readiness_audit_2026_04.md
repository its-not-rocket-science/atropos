# Production Readiness Audit (2026-04-08)
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.


## Scope
Audit focus areas requested:
- reliability
- observability
- failure modes
- reproducibility

Primary modules reviewed:
- `src/atropos/pipeline/runner.py`
- `src/atropos/utils/resilience.py`
- `src/atropos/utils/error_categories.py`
- `src/atropos/deployment/health.py`
- `src/atropos/deployment/strategies.py`
- `src/atropos/telemetry_collector.py`
- `src/atropos/telemetry.py`
- `src/atropos/logging_config.py`
- `src/atropos/model_tester.py`

---

## 1) Prioritized risk list

### P0 — Unbounded external command execution can hang pipeline stages
**Where:** `pipeline/runner.py` (`subprocess.run(..., shell=True, check=True, capture_output=True, text=True)`) in prune/recover/validate/deploy/rollback custom command paths.

**Why this is high risk:**
- No `timeout` is passed, so a stalled external tool can block stage completion indefinitely.
- `shell=True` increases command-injection and quoting fragility risks if commands are user/config supplied.
- `capture_output=True` can accumulate large stdout/stderr in memory for long-running processes.

**Failure mode:** pipeline deadlock/hang; process memory pressure; no deterministic recovery path.

---

### P0 — No retry, backoff, or circuit breaker around deployment-time network checks
**Where:** `deployment/health.py`, `deployment/strategies.py`, `telemetry_collector.py`.

**Why this is high risk:**
- Health checks and telemetry HTTP calls are mostly single-shot requests.
- Transient DNS/TCP/TLS failures and brief 5xx spikes are treated as immediate failures.
- Canary/blue-green loops poll status but do not implement explicit exponential backoff with jitter or circuit open/half-open semantics.

**Failure mode:** false negatives during rollout; unnecessary rollback; noisy incident pages.

---

### P1 — Process isolation exists only in batch path, not in pipeline execution path
**Where:** `utils/resilience.py` is used by `batch.py`, but pipeline stage execution does not use timeout wrappers or isolated worker processes.

**Why this matters:**
- `batch_process` has a robust timeout+retry envelope.
- Core pipeline stage code for external commands/platform calls does not share the same resilience controls.

**Failure mode:** reliability behavior differs by entrypoint; operational surprises between batch and pipeline modes.

---

### P1 — Logging lacks correlation IDs and structured context needed for production tracing
**Where:** `logging_config.py`, plus general logging patterns in pipeline/deployment/telemetry modules.

**Observed gaps:**
- No request/run/deployment correlation ID propagated across modules.
- JSON logging is available but minimal; no standard fields for trace/span IDs, scenario, strategy, stage, retry-attempt, endpoint, error category.
- Many exceptions are logged as plain text without machine-parseable error taxonomy.

**Failure mode:** slow incident triage, poor root-cause analysis, weak SLO debugging.

---

### P1 — Inference/model-loading paths can run long without hard deadline enforcement
**Where:** `model_tester.py` accepts `timeout_sec` but does not enforce it during model load/inference.

**Why this matters:**
- Large model pulls or cold-start loads can stall long enough to block CI/validation workflows.

**Failure mode:** validation jobs exceed budget or never finish under degraded remote model registry/network conditions.

---

### P2 — Telemetry parsing/collection quality issues reduce metric trust
**Where:** `telemetry.py`, `telemetry_collector.py`.

**Examples:**
- Triton parser uses rough/fixed assumptions for time windows and token estimates.
- vLLM/TGI collectors fallback silently and often return zeros for key fields.
- Health/path errors are frequently swallowed into generic failure messages.

**Failure mode:** incorrect baseline metrics lead to wrong ROI and gating decisions.

---

### P2 — Reproducibility controls are not explicit in runtime paths
**Where:** pipeline/deployment/telemetry execution paths.

**Observed gaps:**
- No standardized run manifest capturing exact package versions, model revision/hash, config hash, random seeds, hardware fingerprint, and git SHA.
- External commands may depend on mutable environment state.

**Failure mode:** non-reproducible outcomes between environments and between runs.

---

## 2) Points of failure inventory

### Network failure points
- HTTP health checks (`urllib.request.urlopen`) in deployment checks.
- Collector endpoint calls (`/health`, `/metrics`, `/v1/completions`, `/v2/models/stats`).
- Potential model hub downloads in `model_tester.py` via `transformers` loading.

### Model inference failure points
- Runtime OOM and model load/inference failures in `model_tester.py`.
- Pruning framework integration catches broad exceptions and returns opaque error strings.
- Telemetry benchmark requests can fail and be silently ignored during warmup/sampling.

### Environment/process crash points
- External tooling subprocesses launched by pipeline stages.
- GPU runtime instability and CUDA OOM during model/pruning work.
- Worker subprocess dead exits handled in batch mode, but not consistently in pipeline mode.

---

## 3) Concrete fixes

### A) Retry strategy standards
1. **Adopt per-call retry policy object for all outbound network calls**
   - exponential backoff with jitter (e.g., 100ms base, cap 5s)
   - max attempts by endpoint class:
     - health check reads: 3 attempts
     - deployment control-plane writes: 5 attempts (idempotency key required)
     - telemetry sample reads: 2 attempts
2. **Retry only transient classes**
   - retry: DNS/TCP reset/timeout/HTTP 429/502/503/504
   - do not retry: 400/401/403/404 validation/auth errors
3. **Surface attempt metadata in logs/metrics**
   - `attempt`, `max_attempts`, `backoff_ms`, `error_type`.

### B) Timeout defaults (hard limits)
1. **Subprocess stage commands**
   - add `timeout=` for every `subprocess.run` in pipeline stage code.
   - suggested defaults:
     - prune: 2h
     - recover: 4h
     - validate: 1h
     - deploy/rollback shell ops: 10m
2. **HTTP calls**
   - split connect/read timeout (if library supports); if staying with urllib, enforce conservative total timeout and wrapping watchdog timer.
3. **Global pipeline budget**
   - add overall run deadline and per-stage deadline; fail fast when exceeded.

### C) Circuit breakers
1. **Per-endpoint breaker** for health and telemetry endpoints
   - open after N consecutive transient failures (e.g., 5)
   - half-open after cool-down (e.g., 30s)
   - close after M successful probes (e.g., 2)
2. **Per-platform breaker** in deployment strategy loops
   - avoid hammering unhealthy control planes.
3. **Emit breaker state transition logs/metrics** for on-call visibility.

### D) Reproducibility controls
1. Generate a **run manifest** at pipeline start:
   - git SHA, dirty flag, package lock snapshot, Python version, CUDA version, host/GPU info
   - model IDs + exact revisions
   - effective config + hash
   - random seeds for Python/NumPy/Torch
2. Persist manifest with outputs and include manifest ID in every stage log.
3. For external commands, store fully rendered command + environment whitelist (redacted secrets).

### E) Failure semantics and error contracts
1. Replace broad `except Exception` returns with typed error envelopes (`category`, `recoverable`, `retryable`, `root_cause`).
2. Preserve structured stderr/stdout excerpts (bounded) and exit code for subprocess failures.
3. Align batch and pipeline to shared resilience middleware.

---

## 4) Logging evaluation (what is missing)

### Missing now
- Correlation IDs (`run_id`, `deployment_id`, `experiment_id`, `stage_id`).
- Consistent structured fields (scenario, strategy, stage, endpoint, attempt, timeout_ms, duration_ms, error_category).
- Standardized error payload schema across modules.
- Log volume controls/sampling policy for noisy loops.
- Redaction policy for sensitive command args/env.

### Recommended logging schema (JSON)
Minimum keys in every event:
- `ts`, `level`, `service`, `module`, `event`, `message`
- `run_id`, `stage`, `scenario`, `strategy`
- `attempt`, `duration_ms`, `timeout_ms`
- `error_category`, `exception_type`, `retryable`
- `endpoint` or `command`

---

## 5) Metrics to track in production

### Reliability & failure metrics
- `pipeline_runs_total{status}`
- `pipeline_stage_duration_seconds{stage,status}`
- `pipeline_stage_failures_total{stage,error_category}`
- `retries_total{component,operation,error_category}`
- `timeouts_total{component,operation}`
- `circuit_breaker_state{component,endpoint,state}`

### Network/dependency metrics
- `http_client_requests_total{target,method,status}`
- `http_client_latency_seconds{target,method}` (histogram)
- `dependency_availability{dependency}`

### Inference/workload metrics
- `inference_requests_total{model,variant,status}`
- `inference_latency_seconds{model,variant,quantile}`
- `tokens_generated_total{model,variant}`
- `throughput_tokens_per_second{model,variant}`
- `gpu_memory_bytes{device}` / `gpu_oom_total{device}`

### Reproducibility & quality metrics
- `run_manifest_mismatch_total{field}`
- `config_hash_cardinality` / `model_revision_cardinality`
- `quality_degradation_percent{model,strategy}`
- `roi_prediction_error_percent{model,strategy}`

---

## 6) Suggested observability stack

### Baseline (good default)
- **OpenTelemetry SDK** for traces + metrics instrumentation in Python modules.
- **Prometheus** for metrics scraping/alerting (or OTEL collector -> managed metrics backend).
- **Grafana** for dashboards (pipeline health, stage latency, retry/circuit panels).
- **Loki** (or ELK/OpenSearch) for structured logs.
- **Tempo/Jaeger** for distributed traces.

### Deployment topology
1. App emits OTEL logs/metrics/traces.
2. OTEL Collector performs:
   - batching
   - retry/export buffering
   - attribute enrichment (env, commit SHA)
3. Export paths:
   - metrics -> Prometheus remote write
   - traces -> Tempo/Jaeger
   - logs -> Loki/OpenSearch

### Alerting starter set
- stage failure rate > 2% for 10m
- p95 stage duration breach per stage budget
- consecutive breaker-open events > threshold
- telemetry collection success < 95%
- deployment rollback count > 0 in rolling 1h window

---

## 7) Implementation sequence (30/60/90)

### 0–30 days
- Add timeouts to all pipeline subprocess calls.
- Introduce shared network client wrapper with retry + jitter.
- Add correlation IDs and structured JSON logging fields.

### 31–60 days
- Add circuit breaker utility and integrate deployment health + telemetry collectors.
- Instrument core pipeline stages with metrics and trace spans.
- Add run manifest generation + persistence.

### 61–90 days
- Harmonize batch and pipeline resilience contracts.
- Add SLO dashboards and burn-rate alerts.
- Run game-day tests for dependency outages and model OOM scenarios.
