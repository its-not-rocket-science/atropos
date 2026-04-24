# Platform Acceptance Checklist

This checklist defines minimum acceptance criteria for **platform-core** changes so the codebase does not regress to research-only assumptions.

## Platform-core modules

Treat the following as platform-core surfaces that require release gating:

- `src/atroposlib/api/`
- `src/atroposlib/envs/`
- `src/atroposlib/workers/runtime.py`
- `src/atroposlib/observability.py`
- `src/atroposlib/logging_utils.py`
- `src/atroposlib/plugins/servers.py`
- `src/atroposlib/api/storage.py`

When a pull request touches one or more paths above, release review must include this checklist and all CI platform-core invariant checks must pass.

## durable state

- [ ] Durable store writes are covered by tests (`test_runtime_server_storage.py`, storage adapter tests).
- [ ] Restart scenarios preserve required state and do not duplicate durable records.

## idempotent ingestion

- [ ] Replayed inputs are safe (no duplicate side-effects).
- [ ] API/runtime ingestion paths maintain idempotency contract coverage.

## health/readiness

- [ ] Runtime exposes health/readiness behavior aligned with deployment expectations.
- [ ] Health checks fail closed for critical dependency failures.

## structured logs

- [ ] Logs are structured, parseable, and include stable keys for correlation.
- [ ] Log schema changes are called out in release notes.

## service metrics

- [ ] Core runtime metrics remain emitted and scrapeable.
- [ ] Metric name/label changes are reviewed for dashboard impact.

## integration tests

- [ ] Runtime integration tests validate end-to-end storage/API behavior.
- [ ] CI includes platform-core targeted tests in addition to baseline suite.

## restart recovery

- [ ] Recovery behavior is validated by tests and/or documented drills.
- [ ] Partial failures recover without manual data surgery.

## deployment artifacts

- [ ] Kubernetes manifests and runtime Dockerfiles stay aligned with runtime interfaces.
- [ ] Release package includes deployable artifacts required by operations.

## config validation

- [ ] Runtime profile/config examples remain valid and documented.
- [ ] Invalid config handling remains explicit and test-covered.

## CI enforcement mapping

The following automated checks enforce this checklist:

- `python scripts/check_platform_core_invariants.py`
- `pytest tests/test_runtime_server_storage.py tests/test_api_runtime_e2e_integration.py tests/test_structured_logging.py tests/test_runtime_controller.py tests/test_worker_runtime_observability.py`
- Baseline quality gates (`ruff`, `mypy`, non-integration pytest suite)

A pull request is not release-ready unless all checks above are green.
