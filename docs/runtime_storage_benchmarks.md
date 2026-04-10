# Runtime Storage Benchmarks

## Setup

- Date: 2026-04-10
- Command:
  - `python benchmarks/runtime_storage_benchmark.py --iterations 2000`
- Host: CI/dev container
- Redis benchmark: not executed (no Redis URL provided in environment)

## Results

| Backend | p50 latency (ms) | p95 latency (ms) | Throughput (ops/s) |
|---|---:|---:|---:|
| InMemoryStore | 0.005 | 0.021 | 133,033.0 |
| RedisStore | N/A | N/A | N/A |

## Notes

- In-memory metrics represent single-process baseline performance.
- For production sizing, rerun with a live Redis endpoint:
  - `python benchmarks/runtime_storage_benchmark.py --iterations 10000 --redis-url redis://<host>:6379/0`
- Compare p95 latency and throughput under expected concurrency to determine pod and Redis capacity.
