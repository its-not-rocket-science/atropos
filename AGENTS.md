# AGENTS.md

## Codex completion rules

Before marking any task as complete (including before `git commit` and before opening a PR), Codex must run and pass all required checks:

```bash
ruff check .
ruff format --check .
pytest -m "not integration and not stress" --cov-report=xml:coverage.xml
mypy src
```

If any required check fails, fix the reported issues and rerun the full required check list until everything passes.

Codex must not report completion, create a commit, or open a PR while any required check is failing.
Codex must also not post a final task-complete message until these checks have passed in the current run.

When changing FastAPI request/response handling or endpoint signatures, Codex must also run:

```bash
pytest tests/test_runtime_server_storage.py tests/test_api_runtime_e2e_integration.py tests/test_structured_logging.py
```

and ensure all selected tests pass before completion.

## FastAPI safety rules

When editing FastAPI endpoints, **do not reuse** `Body(...)`, `Header(...)`, `Query(...)`, `Path(...)`, or other `Param` instances across multiple route function parameters (for example via module-level constants).

Always declare these inline per-parameter so each route gets its own independent field metadata and validation behavior.

Also avoid `Header(...)`/`Body(...)` metadata in shared dependency callables that are reused across routes; read headers/body from `Request` inside dependencies when possible to prevent cross-route validation coupling.
