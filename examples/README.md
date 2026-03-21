# Atropos Examples

This directory contains example configurations and usage patterns for Atropos.

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
