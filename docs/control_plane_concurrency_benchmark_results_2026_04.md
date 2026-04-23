# Control-plane concurrency benchmark results (2026-04-23 UTC)

## Command

```bash
PYTHONPATH=src python benchmarks/runtime_concurrency_benchmark.py --workers 24 --requests 600 --duplicate-attempts 400 --cycles 40
```

## Results

- concurrent_group_ingestion: **PASS**
  - success_ratio=1.0000
  - deduplicated_count=0.0000
  - p50_latency_ms=0.0563
  - p95_latency_ms=4.7966
  - throughput_req_s=7026.2083
- duplicate_submission_storm: **PASS**
  - accepted_total=1.0000
  - first_writer_wins=1.0000
  - duplicate_leakage_ratio=0.0000
- worker_backpressure: **PASS**
  - workers_before=32.0000
  - workers_after=6.0000
  - throttling_ratio=0.1875
  - final_rate_limit=0.2000
- api_register_disconnect_cycles: **PASS (skipped_no_fastapi=1.0000)**
  - cycles=40.0000

## Notes

- This environment did not have FastAPI installed, so API-level register/disconnect benchmark cycles were intentionally skipped by the benchmark harness.
- Store-level and queue-pressure concurrency metrics executed normally.
