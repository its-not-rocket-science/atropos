# Model Testing Guide for Atropos Validation

This guide helps you find and test neural networks and LLMs to validate Atropos projections.

## Quick Start: Recommended Test Models

### Small Models (Fast Testing, < 1B params)
| Model | Params | Use Case | Install |
|-------|--------|----------|---------|
| `gpt2` | 124M | General testing | `pip install transformers` |
| `gpt2-medium` | 355M | Medium tests | `pip install transformers` |
| `gpt2-large` | 774M | Larger tests | `pip install transformers` |
| `gpt2-xl` | 1.5B | Pre-Llama scale | `pip install transformers` |
| `facebook/opt-125m` | 125M | Open-source alt | `pip install transformers` |
| `facebook/opt-350m` | 350M | OPT family | `pip install transformers` |
| `bigscience/bloom-560m` | 560M | Multilingual | `pip install transformers` |
| `EleutherAI/pythia-160m` | 160M | Research model | `pip install transformers` |
| `EleutherAI/pythia-410m` | 410M | Research model | `pip install transformers` |

### Code-Specific Models (For Coding LLM scenarios)
| Model | Params | Use Case |
|-------|--------|----------|
| `microsoft/codegpt-small` | 124M | Code completion |
| `microsoft/DialoGPT-small` | 117M | Dialogue systems |
| `Salesforce/codet5-small` | 60M | Code generation |
| `t5-small` | 60M | General seq2seq |
| `t5-base` | 220M | General seq2seq |

### Medium Models (Realistic Testing, 1B-7B params)
| Model | Params | VRAM Required | Notes |
|-------|--------|---------------|-------|
| `meta-llama/Llama-2-7b-hf` | 7B | ~14GB | Requires HF login |
| `TinyLlama/TinyLlama-1.1B-v1.0` | 1.1B | ~2.5GB | Open, no login |
| `stabilityai/stablelm-3b-4e1t` | 3B | ~6GB | Stability AI |
| `HuggingFaceTB/SmolLM-1.7B` | 1.7B | ~3.5GB | Recent, open |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ~8GB | Strong small model |

## Finding Models

### 1. HuggingFace Model Hub
The primary source for open models:
```bash
# Install CLI tool
pip install huggingface-hub

# Search for models
huggingface-cli search gpt2
huggingface-cli search llama

# List popular models
huggingface-cli scan-cache
```

Website: https://huggingface.co/models

**Filters to use:**
- **Library:** PyTorch
- **Task:** Text Generation
- **Size:** < 1B params (for testing)
- **License:** Apache 2.0, MIT, OpenRAIL (for permissive use)

### 2. Model Discovery Script

Create `find_models.py`:
```python
#!/usr/bin/env python3
"""Discover available models for Atropos testing."""

from huggingface_hub import HfApi, ModelFilter
import torch

def find_test_models(max_params=2_000_000_000):
    """Find suitable test models from HuggingFace."""
    api = HfApi()

    # Filter for text generation models
    models = api.list_models(
        filter=ModelFilter(
            task="text-generation",
            library="pytorch",
        ),
        sort="downloads",
        direction=-1,
        limit=100
    )

    print("Available Models for Testing:")
    print("-" * 80)

    for model in models:
        model_id = model.modelId

        # Skip if requires authentication (most LLaMA variants)
        if model.private or "meta-llama" in model_id.lower():
            continue

        # Get parameter count from tags if available
        params = None
        for tag in model.tags or []:
            if tag.endswith("B") and tag[:-1].replace(".", "").isdigit():
                try:
                    params = float(tag[:-1]) * 1e9
                except ValueError:
                    continue

        if params and params <= max_params:
            print(f"{model_id:50s} {params/1e9:>6.1f}B")

        elif not params and any(x in model_id.lower() for x in ["small", "tiny", "mini"]):
            print(f"{model_id:50s}     ?B (likely small)")

if __name__ == "__main__":
    find_test_models()
```

### 3. Check Local Models

```python
# list_local_models.py
from transformers import AutoModel, MODEL_MAPPING

print("Installed Transformers Models:")
print(MODEL_MAPPING.keys())
```

## Running Validation

### Basic Test (Fast, Small Model)
```bash
# Test with GPT-2 (124M params) - very fast
atropos validate edge-coder --model gpt2 --device cpu

# Test with GPT-2 Medium (355M params)
atropos validate medium-coder --model gpt2-medium --device cpu
```

### Realistic Test (Medium Model, GPU)
```bash
# Requires ~2.5GB VRAM
atropos validate medium-coder \
    --model TinyLlama/TinyLlama-1.1B-v1.0 \
    --device cuda \
    --strategy structured_pruning
```

### Batch Testing Multiple Models
```bash
#!/bin/bash
# test_models.sh

MODELS=("gpt2" "gpt2-medium" "facebook/opt-125m")
STRATEGIES=("mild_pruning" "structured_pruning")

for model in "${MODELS[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        echo "Testing $model with $strategy"
        atropos validate edge-coder \
            --model "$model" \
            --strategy "$strategy" \
            --output "results/${model}_${strategy}.md"
    done
done
```

## Benchmark Datasets

### For Code LLMs
1. **HumanEval** (OpenAI)
   - 164 programming problems
   - Install: `pip install datasets`
   - Load: `datasets.load_dataset("openai_humaneval")`

2. **MBPP** (Google)
   - 974 Python programming problems
   - Load: `datasets.load_dataset("mbpp")`

3. **CodeParrot/GitHub-Code** (HuggingFace)
   - Large code corpus
   - Good for throughput testing

### General Text
1. **C4** (Common Crawl)
   - Large text corpus
   - Load: `datasets.load_dataset("c4", "en")`

2. **WikiText-103**
   - Wikipedia articles
   - Load: `datasets.load_dataset("wikitext", "wikitext-103-raw-v1")`

## Testing Workflows

### Workflow 1: Quick Smoke Test
```bash
# 1. Test basic functionality
atropos validate edge-coder --model gpt2

# 2. Check if Atropos is in the right ballpark
# Expect: variance < 20% for memory, < 30% for throughput
```

### Workflow 2: Full Calibration
```bash
# 1. Create a scenario file
cat > my_scenario.yaml <<EOF
name: my-test-model
parameters_b: 0.355
gpu_tier: A100_40GB
memory_gb: 0.7
throughput_toks_per_sec: 50
power_watts: 60
requests_per_day: 10000
tokens_per_request: 100
electricity_cost_per_kwh: 0.15
one_time_project_cost_usd: 5000
batch_size: 1
EOF

# 2. Validate against actual model
atropos validate my_scenario.yaml --model gpt2-medium

# 3. If variance > 20%, calibrate
atropos calibrate my_scenario.yaml telemetry.json
```

### Workflow 3: Pruning Effectiveness Study
```bash
# Test different pruning levels
for sparsity in 0.1 0.2 0.3 0.4; do
    # Create custom strategy
    python -c "
import json
strategy = {
    'name': f'prune_{int(sparsity*100)}',
    'parameter_reduction_fraction': sparsity,
    'memory_reduction_fraction': sparsity * 0.7,
    'throughput_improvement_fraction': sparsity * 0.5,
    'power_reduction_fraction': sparsity * 0.4,
    'quality_risk': 'low' if sparsity < 0.3 else 'medium'
}
print(json.dumps(strategy))
" > strategy.json

    # Run validation
    atropos validate edge-coder \
        --model gpt2 \
        --strategy structured_pruning \
        --output "results/sparsity_${sparsity}.json"
done
```

## Measuring Real Hardware

### CPU Testing
```bash
# Basic CPU test (works everywhere)
atropos validate medium-coder --model gpt2 --device cpu

# Monitor resources
# Linux/Mac:
time -v atropos validate medium-coder --model gpt2

# Or use psrecord:
pip install psrecord
psrecord "atropos validate medium-coder --model gpt2" --plot plot.png
```

### GPU Testing
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run with GPU
atropos validate medium-coder --model gpt2 --device cuda

# Monitor GPU usage
# Install: pip install gpustat
watch -n 1 gpustat
```

## Collecting Results

### Aggregate Results
```python
#!/usr/bin/env python3
# aggregate_results.py
import json
import glob
from pathlib import Path

results = []
for file in glob.glob("results/*.json"):
    with open(file) as f:
        data = json.load(f)
        results.append({
            "model": data["baseline_metrics"]["model_name"],
            "strategy": data["strategy_name"],
            "memory_variance": next(
                (c["variance_pct"] for c in data["comparisons"] if c["name"] == "Memory"),
                None
            ),
            "throughput_variance": next(
                (c["variance_pct"] for c in data["comparisons"] if c["name"] == "Throughput"),
                None
            ),
            "accuracy": data["savings_accuracy"]
        })

# Print summary
print("Model, Strategy, Memory Var %, Throughput Var %, Accuracy")
for r in results:
    print(f"{r['model']}, {r['strategy']}, {r['memory_variance']}, {r['throughput_variance']}, {r['accuracy']}")
```

## Troubleshooting

### Model Won't Load
```bash
# Check model exists
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"

# Clear cache if corrupted
rm -rf ~/.cache/huggingface/hub/models--gpt2
```

### Out of Memory
```bash
# Use smaller batch size (in scenario config)
# Use CPU instead
atropos validate medium-coder --model gpt2 --device cpu

# Or use a smaller model
atropos validate edge-coder --model gpt2
```

### Slow Execution
```bash
# Limit sequence length
export TRANSFORMERS_MAX_LENGTH=128

# Use quantized model
atropos validate medium-coder --model "gpt2"  # No quantization yet
```

## Next Steps

1. **Start Small**: Test with `gpt2` first
2. **Build a Test Suite**: Create YAML scenarios for each model family
3. **Track Over Time**: Re-run validation after Atropos updates
4. **Contribute Results**: Share findings to improve Atropos models
