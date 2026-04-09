# AGENTS.md

## Codex completion rule

Before marking any task as complete, Codex must run and pass:

```bash
ruff check .
```

If `ruff check .` fails, fix the reported issues and rerun the command until it passes.
