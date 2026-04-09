# AGENTS.md

## Codex completion rules

Before marking any task as complete (including before `git commit` and before opening a PR), Codex must run and pass all required checks:

```bash
ruff check .
```

If `ruff check .` fails, fix the reported issues and rerun the command until it passes.

Codex must not report completion, create a commit, or open a PR while any required check is failing.
