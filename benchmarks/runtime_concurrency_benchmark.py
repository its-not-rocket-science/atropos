"""Concurrency benchmark for Atropos runtime and API paths.

Usage:
    PYTHONPATH=src python benchmarks/runtime_concurrency_benchmark.py
    PYTHONPATH=src python benchmarks/runtime_concurrency_benchmark.py --workers 64 --requests 2000
"""

from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.util import find_spec

from atroposlib.api.storage import InMemoryStore, ScoredDataGroup
from atroposlib.envs.worker_runtime import WorkerRuntime

_HAS_FASTAPI = find_spec("fastapi") is not None
if _HAS_FASTAPI:
    from fastapi.testclient import TestClient

    from atroposlib.api.server import build_runtime_app


@dataclass(frozen=True)
class BenchmarkThresholds:
    min_success_ratio: float = 1.0
    max_duplicate_leakage: float = 0.01
    max_p95_latency_ms: float = 75.0
    max_throttling_ratio: float = 0.40


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    passed: bool
    details: dict[str, float]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((percentile / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def _scenario_concurrent_group_ingestion(
    *,
    workers: int,
    requests: int,
    thresholds: BenchmarkThresholds,
) -> BenchmarkResult:
    store = InMemoryStore()
    latencies_ms: list[float] = []

    def _submit(idx: int) -> tuple[int, bool, float]:
        started = time.perf_counter()
        result = store.ingest_scored_data(
            request_id=f"request-{idx}",
            groups=[
                ScoredDataGroup(
                    environment_id=f"bench-{idx % 8}",
                    group_id=f"group-{idx}",
                    records=[
                        {"sample_id": f"sample-{idx}-a", "score": 0.4},
                        {"sample_id": f"sample-{idx}-b", "score": 0.6},
                    ],
                )
            ],
        )
        latency_ms = (time.perf_counter() - started) * 1000
        return result.accepted_groups, result.deduplicated, latency_ms

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(_submit, range(requests)))
    elapsed = time.perf_counter() - started

    accepted_groups = sum(accepted for accepted, _, _ in outcomes)
    deduplicated = sum(1 for _, is_deduped, _ in outcomes if is_deduped)
    latencies_ms = [latency for _, _, latency in outcomes]
    success_ratio = accepted_groups / requests if requests else 0.0
    p50_ms = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95_ms = _percentile(latencies_ms, 95)
    throughput = requests / elapsed if elapsed else 0.0
    passed = (
        success_ratio >= thresholds.min_success_ratio and p95_ms <= thresholds.max_p95_latency_ms
    )

    return BenchmarkResult(
        name="concurrent_group_ingestion",
        passed=passed,
        details={
            "success_ratio": success_ratio,
            "deduplicated_count": float(deduplicated),
            "p50_latency_ms": p50_ms,
            "p95_latency_ms": p95_ms,
            "throughput_req_s": throughput,
        },
    )


def _scenario_duplicate_storm(
    *,
    workers: int,
    attempts: int,
    thresholds: BenchmarkThresholds,
) -> BenchmarkResult:
    store = InMemoryStore()

    def _submit(_: int) -> tuple[bool, int]:
        result = store.ingest_scored_data(
            request_id="dup-request",
            groups=[
                ScoredDataGroup(
                    environment_id="dup-bench",
                    group_id="dup-group",
                    records=[{"sample_id": "same", "score": 0.9}],
                )
            ],
        )
        return result.deduplicated, result.accepted_count

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(_submit, range(attempts)))

    dedupe_false = sum(1 for deduped, _ in outcomes if not deduped)
    accepted_total = sum(accepted for _, accepted in outcomes)
    duplicate_leakage = max(0, dedupe_false - 1) / max(1, attempts)
    passed = (
        accepted_total == 1
        and dedupe_false == 1
        and duplicate_leakage <= thresholds.max_duplicate_leakage
    )

    return BenchmarkResult(
        name="duplicate_submission_storm",
        passed=passed,
        details={
            "accepted_total": float(accepted_total),
            "first_writer_wins": float(dedupe_false),
            "duplicate_leakage_ratio": duplicate_leakage,
        },
    )


def _scenario_backpressure(thresholds: BenchmarkThresholds) -> BenchmarkResult:
    runtime = WorkerRuntime(min_workers=1, max_workers=64, target_queue_depth=8)

    for idx in range(24):
        runtime.enqueue({"idx": idx})

    before = runtime.recommended_workers(requested_workers=32, env="bench-bp")
    for _ in range(24):
        runtime.update_trainer_feedback("bench-bp", {"queue_depth": 200})
    after = runtime.recommended_workers(requested_workers=32, env="bench-bp")

    throttling_ratio = (after / before) if before else 0.0
    passed = throttling_ratio <= thresholds.max_throttling_ratio

    return BenchmarkResult(
        name="worker_backpressure",
        passed=passed,
        details={
            "workers_before": float(before),
            "workers_after": float(after),
            "throttling_ratio": throttling_ratio,
            "final_rate_limit": runtime.environment_rates["bench-bp"],
        },
    )


def _scenario_api_register_disconnect_cycles(cycles: int) -> BenchmarkResult:
    if not _HAS_FASTAPI:
        return BenchmarkResult(
            name="api_register_disconnect_cycles",
            passed=True,
            details={"skipped_no_fastapi": 1.0, "cycles": float(cycles)},
        )

    ready_ok = 0
    write_ok = 0
    started = time.perf_counter()
    for cycle in range(cycles):
        app = build_runtime_app(store=InMemoryStore())
        with TestClient(app) as client:
            ready = client.get("/health/ready")
            write = client.post(
                "/scored_data",
                json={
                    "environment_id": f"bench-cycle-{cycle % 5}",
                    "group_id": f"group-{cycle}",
                    "records": [{"sample_id": f"sample-{cycle}", "score": 0.2}],
                },
                headers={"X-Request-ID": f"bench-{cycle}"},
            )
            ready_ok += int(ready.status_code == 200)
            write_ok += int(write.status_code == 200)

    elapsed = time.perf_counter() - started
    passed = ready_ok == cycles and write_ok == cycles
    return BenchmarkResult(
        name="api_register_disconnect_cycles",
        passed=passed,
        details={
            "ready_success_ratio": ready_ok / cycles,
            "write_success_ratio": write_ok / cycles,
            "cycles_per_second": (cycles / elapsed) if elapsed else 0.0,
        },
    )


def run_benchmark(args: argparse.Namespace) -> list[BenchmarkResult]:
    thresholds = BenchmarkThresholds(
        min_success_ratio=args.min_success_ratio,
        max_duplicate_leakage=args.max_duplicate_leakage,
        max_p95_latency_ms=args.max_p95_latency_ms,
        max_throttling_ratio=args.max_throttling_ratio,
    )
    return [
        _scenario_concurrent_group_ingestion(
            workers=args.workers,
            requests=args.requests,
            thresholds=thresholds,
        ),
        _scenario_duplicate_storm(
            workers=args.workers,
            attempts=args.duplicate_attempts,
            thresholds=thresholds,
        ),
        _scenario_backpressure(thresholds=thresholds),
        _scenario_api_register_disconnect_cycles(cycles=args.cycles),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--requests", type=int, default=1200)
    parser.add_argument("--duplicate-attempts", type=int, default=800)
    parser.add_argument("--cycles", type=int, default=80)
    parser.add_argument("--min-success-ratio", type=float, default=1.0)
    parser.add_argument("--max-duplicate-leakage", type=float, default=0.0)
    parser.add_argument("--max-p95-latency-ms", type=float, default=10.0)
    parser.add_argument("--max-throttling-ratio", type=float, default=0.45)
    args = parser.parse_args()

    results = run_benchmark(args)
    any_failed = False
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        metrics = " ".join(f"{key}={value:.4f}" for key, value in result.details.items())
        print(f"{result.name:<32} {status} {metrics}")
        if not result.passed:
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
