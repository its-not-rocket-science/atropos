## Summary

<!-- Describe what changed and why. -->

## Platform-core modules touched

- [ ] `src/atroposlib/api/`
- [ ] `src/atroposlib/envs/`
- [ ] `src/atroposlib/workers/runtime.py`
- [ ] `src/atroposlib/observability.py`
- [ ] `src/atroposlib/logging_utils.py`
- [ ] `src/atroposlib/plugins/servers.py`
- [ ] `src/atroposlib/api/storage.py`
- [ ] None of the above

> If any platform-core module is touched, complete the acceptance checklist below and do not merge/release until all platform-core invariant checks pass in CI.

## Platform acceptance checklist

### durable state
- [ ] Verified

### idempotent ingestion
- [ ] Verified

### health/readiness
- [ ] Verified

### structured logs
- [ ] Verified

### service metrics
- [ ] Verified

### integration tests
- [ ] Verified

### restart recovery
- [ ] Verified

### deployment artifacts
- [ ] Verified

### config validation
- [ ] Verified

## Validation

- [ ] `python scripts/check_platform_core_invariants.py`
- [ ] `pytest tests/test_runtime_server_storage.py tests/test_api_runtime_e2e_integration.py tests/test_structured_logging.py tests/test_runtime_controller.py tests/test_worker_runtime_observability.py`
- [ ] Baseline CI checks (`ruff`, `mypy`, non-integration test suite)
