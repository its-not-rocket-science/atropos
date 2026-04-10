"""Micro-benchmarks for runtime storage backends.

Usage:
    PYTHONPATH=src python benchmarks/runtime_storage_benchmark.py --iterations 5000
    PYTHONPATH=src python benchmarks/runtime_storage_benchmark.py --iterations 5000 --redis-url redis://localhost:6379/0
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atroposlib.api.storage import RuntimeStore

_STORAGE_SPEC = importlib.util.spec_from_file_location(
    "runtime_storage_module",
    Path(__file__).resolve().parents[1] / "src" / "atroposlib" / "api" / "storage.py",
)
if _STORAGE_SPEC is None or _STORAGE_SPEC.loader is None:
    raise RuntimeError("Unable to load storage module for benchmarks")
_storage_module = importlib.util.module_from_spec(_STORAGE_SPEC)
sys.modules[_STORAGE_SPEC.name] = _storage_module
_STORAGE_SPEC.loader.exec_module(_storage_module)
InMemoryStore = _storage_module.InMemoryStore
RedisStore = _storage_module.RedisStore


def _benchmark_store(store: RuntimeStore, iterations: int) -> dict[str, float]:
    durations_ms: list[float] = []
    started = time.perf_counter()
    for idx in range(iterations):
        t0 = time.perf_counter()
        store.enqueue_job(
            job_id=f"job-{idx}",
            now=datetime.now(tz=timezone.utc),
            idempotency_key=f"idem-{idx}",
        )
        durations_ms.append((time.perf_counter() - t0) * 1000)
    elapsed = time.perf_counter() - started
    throughput = iterations / elapsed if elapsed else 0.0
    return {
        "p50_ms": statistics.median(durations_ms),
        "p95_ms": statistics.quantiles(durations_ms, n=20)[18],
        "throughput_ops_s": throughput,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--redis-url", type=str, default=None)
    args = parser.parse_args()

    memory = _benchmark_store(InMemoryStore(), args.iterations)
    print(
        "in_memory "
        f"p50_ms={memory['p50_ms']:.3f} "
        f"p95_ms={memory['p95_ms']:.3f} "
        f"throughput={memory['throughput_ops_s']:.1f}"
    )

    if args.redis_url:
        redis = _benchmark_store(RedisStore.from_url(args.redis_url), args.iterations)
        print(
            "redis     "
            f"p50_ms={redis['p50_ms']:.3f} "
            f"p95_ms={redis['p95_ms']:.3f} "
            f"throughput={redis['throughput_ops_s']:.1f}"
        )
    else:
        print("redis     skipped (no --redis-url provided)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
