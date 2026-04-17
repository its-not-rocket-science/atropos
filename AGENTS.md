# AGENTS.md

## Codex completion rules

Before marking any task as complete (including before `git commit` and before opening a PR), Codex must run and pass all required checks:

```bash
ruff check .
ruff format --check .
pytest -m "not integration and not stress" --cov-report=xml:coverage.xml
```

If any required check fails, fix the reported issues and rerun the full required check list until everything passes.

Codex must not report completion, create a commit, or open a PR while any required check is failing.
