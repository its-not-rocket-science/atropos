# Atropos Examples

## Minimal example set (first run)

If you are new to Atropos, start here first:

```bash
python examples/minimal/toy_env.py
python examples/minimal/toy_trainer_walkthrough.py
```

Files:
- `examples/minimal/toy_env.py`: tiny deterministic environment (`reset` + `step`).
- `examples/minimal/toy_trainer_walkthrough.py`: tiny trainer loop that builds a trajectory report.

## Pipeline Configuration

The `pipeline-config.yaml` file demonstrates a complete Atropos Pipeline configuration for automated pruning and tuning.

### Usage

Validate the configuration:
```bash
atropos-llm validate-pipeline-config examples/pipeline-config.yaml
```

Run the pipeline (dry-run mode):
```bash
atropos-llm pipeline medium-coder \
    --config examples/pipeline-config.yaml \
    --strategy structured_pruning \
    --dry-run
```

Run the pipeline for real:
```bash
atropos-llm pipeline medium-coder \
    --config examples/pipeline-config.yaml \
    --strategy structured_pruning \
    --output results.json
```

### Pipeline Stages

1. **Assess** - Run ROI analysis on the scenario
2. **Gate** - Check if projected savings meet thresholds
3. **Prune** - Execute pruning using configured framework
4. **Recover** - Fine-tune the pruned model (if enabled)
5. **Validate** - Benchmark and verify quality
6. **Deploy** - Deploy the optimized model (if auto_deploy enabled)

### Configuration Options

See the full configuration reference in `pipeline-config.yaml`.

Key settings:
- `thresholds`: Define minimum ROI requirements
- `pruning.framework`: Choose from llm-pruner, wanda, sparsegpt, or custom
- `recovery.enabled`: Enable/disable fine-tuning
- `deployment.auto_deploy`: Enable automatic deployment after validation


## Runtime deployment profiles

Sample `.env` files for runtime API configuration layers:

- `examples/runtime-profiles/local-dev.env`
- `examples/runtime-profiles/ci.env`
- `examples/runtime-profiles/production.env.example`
