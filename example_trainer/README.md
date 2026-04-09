# example_trainer

This path is maintained for documentation compatibility.

There is no dedicated `example_trainer` package in the current codebase. For runnable examples and pipeline usage, use:

- `examples/README.md`
- `examples/pipeline-config.yaml`

Primary CLI commands for examples:

```bash
atropos-llm validate-pipeline-config examples/pipeline-config.yaml
atropos-llm pipeline medium-coder --config examples/pipeline-config.yaml --strategy structured_pruning --dry-run
```
