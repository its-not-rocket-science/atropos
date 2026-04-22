# Runtime storage backend contract tests

This project includes a **backend contract suite** that validates the shared `AtroposStore` behavior across all storage implementations.

Current backend adapters:

- `InMemoryStore`
- `RedisStore`

Contract tests live in:

- `tests/test_api_store_backends.py`
- adapter definitions: `tests/contracts_store_adapters.py`

## Behavioral expectations covered

The suite verifies the core store-layer contract for:

1. **enqueue**
   - first enqueue inserts a queued job
   - queue depth increments correctly
   - `get_job_status()` returns queued metadata
2. **dedupe**
   - idempotency key deduplicates duplicate `enqueue_job` requests
   - repeated scored-data request/group ingestion is deduplicated
3. **buffer**
   - ingestion transitions through buffered lifecycle metadata
4. **batch selection**
   - `list_scored_data(..., limit=N)` returns a bounded batch slice
5. **env registration**
   - explicit environment registration is idempotent
   - registered environment inventory can be listed
   - unknown environments return empty results
   - ingested records are isolated per `environment_id`
6. **step/status updates**
   - group lifecycle reaches `acknowledged`
   - lifecycle timestamps are set and monotonic
7. **durability semantics**
   - durable backends recover state across adapter-driven restart
   - non-durable backends do not persist across restart

## Adding a future backend to the contract suite

When a new store implementation is added, wire it into `tests/contracts_store_adapters.py`:

1. Add a `BackendAdapter` factory that can:
   - build a fresh store instance (`build`)
   - build a restarted store against the same underlying persistence (`restart`)
   - declare durability (`durable=True/False`)
2. Add the adapter name to `CONTRACT_BACKEND_NAMES` and handle it in `create_backend_adapter(...)`.
3. Run the required checks:

```bash
ruff check .
ruff format --check .
pytest -m "not integration and not stress" --cov-report=xml:coverage.xml
```

If the backend is durable, restart assertions must pass with persisted state.
If the backend is process-local/non-durable, restart assertions must show empty recovered state.
