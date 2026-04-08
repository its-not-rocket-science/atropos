# Model Recommendations for Atropos
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.


This guide helps you select appropriate models for testing and validating Atropos projections based on your deployment scenario.

## Quick Reference

| Deployment | Size | Example Models | Atropos Preset |
|------------|------|----------------|----------------|
| Edge/Embedded | < 1B | gpt2, opt-125m, pythia-160m | `edge-coder` |
| Small Server | 1-3B | gpt2-xl, TinyLlama-1.1B | `small-coder` |
| Medium Server | 3-7B | opt-2.7b, pythia-2.8b | `medium-coder` |
| Large Server | 7-13B | Llama-2-7b, Mistral-7B | `large-coder` |
| Enterprise | > 13B | Llama-2-13b, Mixtral-8x7B | Custom |

## Model Size Tiers

### Edge Models (< 1B parameters)

Best for: Mobile apps, IoT devices, browser-based inference, low-latency edge deployments.

| Model | Params | Memory* | License | Use Case |
|-------|--------|---------|---------|----------|
| **gpt2** | 124M | 0.5 GB | MIT | General text, prototyping |
| **facebook/opt-125m** | 125M | 0.5 GB | Apache 2.0 | Research, experimentation |
| **EleutherAI/pythia-160m** | 160M | 0.6 GB | Apache 2.0 | Research, interpretability |
| **gpt2-medium** | 355M | 1.4 GB | MIT | Better quality edge |
| **facebook/opt-350m** | 350M | 1.4 GB | Apache 2.0 | Research, fine-tuning |
| **EleutherAI/pythia-410m** | 410M | 1.6 GB | Apache 2.0 | Research tasks |
| **bigscience/bloom-560m** | 560M | 2.2 GB | BigScience RAIL | Multilingual |
| **gpt2-large** | 774M | 3.1 GB | MIT | High-quality edge |

*Memory estimates assume FP32. Use FP16/BF16 for ~50% reduction.

**Recommended for Atropos testing:**
- Start with `gpt2` (fastest load, widely compatible)
- Use `gpt2-medium` for more realistic edge scenarios
- Try `pythia-160m` for research/interpretability work

### Medium Models (1-7B parameters)

Best for: Small to medium server deployments, cost-effective cloud inference, startup workloads.

| Model | Params | Memory* | License | Strengths |
|-------|--------|---------|---------|-----------|
| **gpt2-xl** | 1.5B | 6.0 GB | MIT | Reliable baseline |
| **TinyLlama/TinyLlama-1.1B** | 1.1B | 4.4 GB | Apache 2.0 | Modern architecture, efficient |
| **bigscience/bloom-1b7** | 1.7B | 6.8 GB | BigScience RAIL | Multilingual |
| **facebook/opt-1.3b** | 1.3B | 5.2 GB | Apache 2.0 | Research, reproducibility |
| **EleutherAI/pythia-1.4b** | 1.4B | 5.6 GB | Apache 2.0 | Training stability research |
| **EleutherAI/pythia-2.8b** | 2.8B | 11.2 GB | Apache 2.0 | Mid-size research |
| **openlm-research/open_llama_3b** | 3B | 12.0 GB | Apache 2.0 | Llama architecture |
| **stabilityai/stablelm-3b-4e1t** | 3B | 12.0 GB | CC BY-SA-4.0 | Long context |

*Memory estimates assume FP32. Use FP16/BF16 for ~50% reduction.

**Recommended for Atropos testing:**
- `gpt2-xl` as a baseline (widely tested)
- `TinyLlama-1.1B` for modern efficiency comparisons
- `pythia-2.8b` for research on scaling effects

### Large Models (7-13B parameters)

Best for: Production APIs, enterprise deployments, high-quality applications.

| Model | Params | Memory* | License | Notes |
|-------|--------|---------|---------|-------|
| **meta-llama/Llama-2-7b** | 7B | 28 GB | Llama 2 License | Requires approval |
| **mistralai/Mistral-7B-v0.1** | 7B | 28 GB | Apache 2.0 | High performance |
| **HuggingFaceTB/SmolLM-1.7B** | 1.7B | 6.8 GB | Apache 2.0 | Actually medium, but efficient |
| **microsoft/Phi-3-mini-4k-instruct** | 3.8B | 15 GB | MIT | Strong small model |
| **tiiuae/falcon-7b** | 7B | 28 GB | Apache 2.0 | RefinedWeb training |
| **mosaicml/mpt-7b** | 7B | 28 GB | Apache 2.0 | ALiBi, FlashAttention |

*Memory estimates assume FP16. FP32 would be 2x.

**Note:** Many 7B+ models require:
- HuggingFace login: `huggingface-cli login`
- Model access approval (Llama 2, some others)
- Significant GPU memory (16GB+ for FP16, 32GB+ for FP32)

## Code-Specific Models

For code completion and generation scenarios:

| Model | Params | Language | Best For |
|-------|--------|----------|----------|
| **microsoft/codegpt-small** | 124M | Python | Entry-level code completion |
| **Salesforce/codet5-small** | 60M | Multi-language | Code translation, summarization |
| **t5-small** | 60M | Multi-language | General seq2seq |
| **t5-base** | 220M | Multi-language | Stronger seq2seq |
| **bigcode/starcoder2-3b** | 3B | 600+ languages | Code generation |
| **microsoft/Phi-3-mini-4k-instruct** | 3.8B | Multi-language | Code + general |
| **codellama/CodeLlama-7b-hf** | 7B | Code | Specialized code model |

## Testing Model Loading

Use the built-in model tester to verify models work on your hardware:

```bash
# Test recommended models on CPU
atropos-llm test-models --device cpu --max-params 1.0

# Test specific models on GPU
atropos-llm test-models \
    --device cuda \
    --models gpt2 gpt2-medium facebook/opt-125m \
    --max-params 1.0 \
    --catalog models-catalog.yaml

# Test a wide range and generate catalog
atropos-llm test-models \
    --device cuda \
    --max-params 7.0 \
    --output test-results.json \
    --catalog models-catalog.yaml
```

## Selecting Models for Your Scenario

### Decision Flowchart

1. **What's your deployment target?**
   - Mobile/edge → < 1B models
   - Small server (1-2 GPUs) → 1-7B models
   - Large server/cluster → 7B+ models

2. **What's your latency requirement?**
   - < 100ms → < 1B models or heavy optimization
   - < 500ms → 1-3B models
   - < 1s → 3-7B models
   - Batch/async → Any size

3. **What's your quality requirement?**
   - Research/prototyping → Any model
   - Production MVP → 1-3B with fine-tuning
   - Production quality → 7B+ or optimized 3B

4. **License constraints?**
   - Commercial use → Apache 2.0, MIT models
   - Research only → Academic licenses
   - Check specific model licenses before deployment

### Example Selections

#### Scenario: Startup MVP
- **Budget:** Limited
- **Hardware:** Single A10G (24GB)
- **Goal:** Deploy code assistant
- **Recommendation:** `bigcode/starcoder2-3b` or `microsoft/Phi-3-mini-4k-instruct`
- **Atropos preset:** `medium-coder`

#### Scenario: Enterprise API
- **Budget:** High
- **Hardware:** A100 cluster
- **Goal:** General-purpose LLM API
- **Recommendation:** `mistralai/Mistral-7B-v0.1` (Apache 2.0)
- **Atropos preset:** `large-coder`

#### Scenario: Edge Device
- **Budget:** Minimal
- **Hardware:** CPU or Jetson
- **Goal:** On-device inference
- **Recommendation:** `TinyLlama/TinyLlama-1.1B-v1.0`
- **Atropos preset:** `edge-coder`

## Hardware Requirements

### Minimum Requirements by Model Size

| Model Size | CPU RAM | GPU VRAM (FP32) | GPU VRAM (FP16) |
|------------|---------|-----------------|-----------------|
| < 500M | 4 GB | 2 GB | 1 GB |
| 500M - 1B | 8 GB | 4 GB | 2 GB |
| 1B - 3B | 16 GB | 12 GB | 6 GB |
| 3B - 7B | 32 GB | 28 GB | 14 GB |
| 7B - 13B | 64 GB | 52 GB | 26 GB |

### Recommended GPUs

| Model Size | Entry | Good | Optimal |
|------------|-------|------|---------|
| < 1B | CPU | GTX 1660 | RTX 3060 |
| 1-3B | RTX 3060 | RTX 3090 | A10G |
| 3-7B | RTX 3090 | A10G | A100 40GB |
| 7-13B | A10G | A100 40GB | A100 80GB |

## Validating with Atropos

Once you've selected models, validate your projections:

```bash
# 1. Test the model loads
atropos-llm test-models --models your-model --device cuda

# 2. Create scenario from model
atropos-llm scenario models-catalog.yaml --strategy structured_pruning

# 3. Run validation against real model
atropos-llm validate your-scenario --model your-model --device cuda

# 4. Calibrate if needed
atropos-llm calibrate your-scenario.yaml telemetry.json
```

## Contributing Tested Models

If you test a model successfully:

1. Run the test suite:
   ```bash
   atropos-llm test-models --models your-model --catalog my-catalog.yaml
   ```

2. Share your catalog entry with:
   - Model ID
   - Hardware used
   - Load time
   - Inference latency
   - Memory footprint

3. Submit to the Atropos model registry (if public)

## Troubleshooting

### Out of Memory Errors

```bash
# Test smaller models first
atropos-llm test-models --max-params 1.0

# Use CPU offloading
atropos-llm test-models --device cpu --max-params 3.0

# Test with quantization (if supported)
# Add --load-in-8bit or --load-in-4bit if implemented
```

### Slow Loading

- Use models with `safetensors` format (faster loading)
- Cache models locally: `export HF_HOME=/fast/storage`
- Use smaller variants first

### Access Denied (Gated Models)

```bash
# Login to HuggingFace
huggingface-cli login

# Request access on model page
# Wait for approval (usually instant for most models)
```

## See Also

- [Model Testing Guide](model-testing-guide.md) - Detailed testing procedures
- [Telemetry Collection Guide](telemetry-collection-guide.md) - Measuring real performance
- [CLI Reference](cli-reference.md) - All Atropos commands
