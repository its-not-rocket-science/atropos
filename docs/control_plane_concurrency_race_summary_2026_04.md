# Atropos control-plane concurrency race summary (2026-04-23 UTC)

## Scope and scenarios

The new test suite `tests/control_plane_integration/test_control_plane_concurrency.py` targets:

1. many concurrent environment submissions,
2. simultaneous environment register/disconnect cycles,
3. mixed group sizes under pressure,
4. repeated duplicate retries,
5. batch consumers polling while ingestion spikes,
6. store reconnect behavior during load.

## Discovered correctness risk

### Risk: ingestion success path could fail on transient metrics-store errors

- In `build_runtime_app`, ingestion endpoints call `_record_ingestion_metrics` after a successful `ingest_scored_data` write.
- `_record_ingestion_metrics` previously called store methods (`get_scored_group_status`, `get_scored_queue_metrics`) without fault isolation.
- During transient reconnect windows, these observability reads can raise exceptions and convert an otherwise-successful write into a failed API response.

## Mitigation implemented

- `_record_ingestion_metrics` is now best-effort and returns early on store-observability exceptions.
- Write-path durability and idempotency decisions remain owned by store ingestion, while metrics collection no longer blocks successful writes.
- This aligns with control-plane safety expectations: telemetry should degrade gracefully, not cause data-path failures.

## Remaining operational guidance

- Run the new control-plane concurrency suite in an environment with FastAPI installed to fully exercise API-level behavior.
- Keep idempotency keys stable for client retries; duplicate suppression correctness depends on consistent request identifiers.
- Continue monitoring queue-metric scan costs for durable stores under very large key cardinality.
