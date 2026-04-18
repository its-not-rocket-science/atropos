# Runtime concurrency/load test summary

Last updated: 2026-04-17 (UTC)

## Scope

This suite targets the pre-production failure modes called out for runtime and API scaling:

1. Many concurrent environment workers submitting groups.
2. Partial-batch pressure (mixed valid/invalid groups).
3. Duplicate submission storms.
4. Backpressure behavior under full trainer queues.
5. Rapid environment register/disconnect cycles.

## Stress suite

Run:

```bash
pytest -m stress tests/test_runtime_api_load_stress.py
```

Pass/fail criteria implemented in tests:

- **Concurrent group ingestion:** all requests must return HTTP 200 and zero unintended deduplication.
- **Partial-batch pressure:** status must be `partial_failed` with exact accepted/failed group accounting.
- **Duplicate storms:** exactly one accepted write, all other writes deduplicated.
- **Backpressure:** worker recommendation must drop materially under sustained queue depth pressure and respect configured floor/ceiling.
- **Register/disconnect cycles:** every cycle must produce healthy readiness and successful ingest before disconnect.

## Benchmark script

Run:

```bash
PYTHONPATH=src python benchmarks/runtime_concurrency_benchmark.py \
  --workers 24 \
  --requests 600 \
  --duplicate-attempts 400 \
  --cycles 40
```

### Thresholds (default)

- `success_ratio >= 1.0` for concurrent ingestion.
- `p95_latency_ms <= 10.0` for in-memory concurrent ingestion.
- `duplicate_leakage_ratio <= 0.0` (no duplicate acceptance beyond first writer).
- `throttling_ratio <= 0.45` under sustained queue pressure.
- API register/disconnect cycles must be all-success when FastAPI is installed; otherwise scenario is skipped and reported.

### Observed run in this environment

```
concurrent_group_ingestion       PASS success_ratio=1.0000 deduplicated_count=0.0000 p50_latency_ms=0.0468 p95_latency_ms=0.0911 throughput_req_s=7697.1326
duplicate_submission_storm       PASS accepted_total=1.0000 first_writer_wins=1.0000 duplicate_leakage_ratio=0.0000
worker_backpressure              PASS workers_before=32.0000 workers_after=6.0000 throttling_ratio=0.1875 final_rate_limit=0.2000
api_register_disconnect_cycles   PASS skipped_no_fastapi=1.0000 cycles=40.0000
```

## Observed bottlenecks and race-risk hotspots

1. **Global lock contention in `InMemoryStore`:** both enqueue and scored-data ingest paths serialize on a single process lock, so p95 latency will rise non-linearly as thread count increases.
2. **Queue-metrics scan cost in durable backends:** `get_scored_queue_metrics` scans all group status keys, which can create O(n) pressure under high cardinality.
3. **Request-level dedupe granularity:** duplicate storms with the same `request_id` are controlled well, but callers must ensure stable request IDs; otherwise retries can bypass dedupe.
4. **Backpressure depends on trainer feedback cadence:** throttling only adjusts when queue-depth feedback is provided, so missing/slow feedback delays worker downshift.
5. **API lifecycle churn:** readiness and ingest across repeated client lifecycle cycles are stable in stress tests; production risk shifts to dependency startup latency (e.g., Redis/network) rather than in-process state.

## Recommended production gate

Require all of the following before release:

- Stress suite green in CI (`pytest -m stress ...`).
- Benchmark script exits 0 with default thresholds on target hardware profile.
- Track trendlines for p95 latency and throttling ratio between commits; block regressions >20% unless justified.
