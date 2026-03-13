# Pruning Exercise Scripts

These scripts support the "Pruning Exercise" roadmap item - executing actual pruning on real models to validate Atropos ROI projections.

## Candidate Models

Five models selected for testing (small to medium size):

| Model | Params | Preset | License |
|-------|--------|--------|---------|
| gpt2 | 124M | edge-coder | MIT |
| gpt2-medium | 355M | small-coder | MIT |
| gpt2-xl | 1.5B | medium-coder | MIT |
| facebook/opt-1.3b | 1.3B | medium-coder | Apache-2.0 |
| EleutherAI/pythia-2.8b | 2.8B | large-coder | Apache-2.0 |

## Scripts

### 1. download_test_models.py

Downloads candidate models to a local cache for testing.

```bash
# Download all candidates
python scripts/download_test_models.py

# Download to specific directory
python scripts/download_test_models.py --test-data-dir ./my_models

# Download specific models
python scripts/download_test_models.py --models gpt2 gpt2-medium

# Use GPU for testing downloads
python scripts/download_test_models.py --device cuda
```

### 2. project_savings.py

Generates Atropos baseline projections for all models and strategies.

```bash
# Generate projections for all models
python scripts/project_savings.py

# Include quantization bonus
python scripts/project_savings.py --with-quantization

# Custom output paths
python scripts/project_savings.py --output results/projections.json --markdown results/projections.md
```

### 3. prune_models.py

Applies magnitude-based pruning to candidate models using PyTorch.

```bash
# Prune all models with all strategies
python scripts/prune_models.py

# Prune specific models
python scripts/prune_models.py --models gpt2 gpt2-medium

# Specific strategies only
python scripts/prune_models.py --strategies mild_pruning

# Custom output directory
python scripts/prune_models.py --output-dir ./my_pruned_models
```

### 4. compare_projections.py

Compares Atropos projected savings vs actual pruning results.

```bash
# Generate comparison report
python scripts/compare_projections.py

# Custom paths
python scripts/compare_projections.py \
    --projections results/projections.json \
    --pruning results/pruning_report.json \
    --output results/comparison.json \
    --markdown results/comparison.md
```

### 5. test_pruning_candidates.py

Runs Atropos validation on downloaded models.

```bash
# Test all candidates with default settings
python scripts/test_pruning_candidates.py

# Test on GPU
python scripts/test_pruning_candidates.py --device cuda

# Test specific models
python scripts/test_pruning_candidates.py --models gpt2 gpt2-medium

# Custom output path
python scripts/test_pruning_candidates.py --output results/my_results.json
```

### 6. generate_case_study.py

Generates comprehensive case study report combining all data sources.

```bash
# Generate case study with default paths
python scripts/generate_case_study.py

# Custom paths
python scripts/generate_case_study.py \
    --projections results/projections.json \
    --pruning results/pruning_report.json \
    --benchmark results/benchmark_report.json \
    --output results/case_study.json \
    --markdown results/case_study.md
```

### 7. validate_pruned_models.py

Validates that pruned models maintain performance vs original models.

**Prerequisites:** Both original and pruned models must exist in `test_data/`.

```bash
# Validate all pruned models
python scripts/validate_pruned_models.py

# Custom test data directory
python scripts/validate_pruned_models.py --test-data-dir ./my_models

# Specific models only
python scripts/validate_pruned_models.py --models gpt2 opt-1.3b

# Use GPU for faster validation
python scripts/validate_pruned_models.py --device cuda
```

**Validation Criteria:**
- Perplexity increase ≤ 20%
- Generation similarity ≥ 70%
- Perplexity stays below 50 (for small models)

**Output:**
- `test_data/validation_report.json` — Detailed metrics per model
- `test_data/validation_report.md` — Human-readable pass/fail report

### 8. prune_wanda.py

Prunes models using **Wanda** (Pruning by Weights AND Activations).

Wanda considers both weight magnitudes and input activation norms for per-output pruning decisions. This often achieves better quality than magnitude-based pruning alone.

Reference: [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)

**Setup:**
```bash
# Clone Wanda repo (if not already done)
git clone --depth 1 https://github.com/locuslab/wanda.git external/wanda
```

**Usage:**
```bash
# Prune gpt2 with 30% sparsity
python scripts/prune_wanda.py --model gpt2 --sparsity 0.3

# Prune with custom settings
python scripts/prune_wanda.py \
    --model gpt2-medium \
    --sparsity 0.5 \
    --device cuda \
    --nsamples 256
```

**Output:**
- `test_data/pruned_models/{model}_wanda/` — Pruned model
- `test_data/wanda_report.json` — Pruning results

### 9. prune_sparsegpt.py

Prunes models using **SparseGPT** method.

SparseGPT is a layer-wise pruning method with reconstruction that maintains model quality better than simple magnitude pruning, especially for GPT models.

Reference: [SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)

**Setup:**
```bash
# Clone Wanda repo (includes SparseGPT)
git clone --depth 1 https://github.com/locuslab/wanda.git external/wanda
```

**Usage:**
```bash
# Prune gpt2 with 30% sparsity
python scripts/prune_sparsegpt.py --model gpt2 --sparsity 0.3

# Prune with custom settings
python scripts/prune_sparsegpt.py \
    --model gpt2-medium \
    --sparsity 0.5 \
    --device cuda \
    --nsamples 256
```

**Output:**
- `test_data/pruned_models/{model}_sparsegpt/` — Pruned model
- `test_data/sparsegpt_report.json` — Pruning results

### 10. prune_advanced_frameworks.py

Batch pruning script that runs both Wanda and SparseGPT on multiple models.

**Usage:**
```bash
# Run all methods on default models (gpt2, gpt2-medium)
python scripts/prune_advanced_frameworks.py

# Custom models
python scripts/prune_advanced_frameworks.py --models gpt2 gpt2-xl

# Specific sparsity levels
python scripts/prune_advanced_frameworks.py --sparsity-levels 0.3 0.5 0.7

# Only Wanda
python scripts/prune_advanced_frameworks.py --methods wanda

# Only SparseGPT
python scripts/prune_advanced_frameworks.py --methods sparsegpt
```

**Output:**
- `test_data/advanced_pruning_report.json` — Complete comparison results
- `test_data/pruned_models/{model}_wanda/` — Wanda pruned models
- `test_data/pruned_models/{model}_sparsegpt/` — SparseGPT pruned models

### 11. upload_to_huggingface.py

Uploads pruned models to HuggingFace Hub.

**Prerequisites:**
1. Create a HuggingFace account at https://huggingface.co/join
2. Get an access token at https://huggingface.co/settings/tokens
3. Login via CLI or set environment variable

```bash
# Install dependencies
pip install huggingface-hub

# Login to HuggingFace (interactive)
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token_here

# Upload to personal account
python scripts/upload_to_huggingface.py

# Upload to organization
python scripts/upload_to_huggingface.py --org your-org-name

# Upload as private repositories
python scripts/upload_to_huggingface.py --private
```

**Note:** Total upload size is ~25 GB (8 pruned models). Ensure you have
disk space and bandwidth for the upload.

### 8. discover-models.py (existing)

Lists and tests available models from HuggingFace.

```bash
# List recommended models
python scripts/discover-models.py --list

# Test model loading
python scripts/discover-models.py --test

# Full workflow
python scripts/discover-models.py --full
```

## Workflow

1. **Download models** (one-time):
   ```bash
   python scripts/download_test_models.py
   ```

2. **Generate baseline projections**:
   ```bash
   python scripts/project_savings.py
   ```

3. **Prune models**:
   ```bash
   python scripts/prune_models.py
   ```

4. **Run validation tests** (after pruning):
   ```bash
   python scripts/test_pruning_candidates.py
   ```

5. **Compare results**:
   ```bash
   python scripts/compare_projections.py
   ```

6. **Run quality benchmarks**:
   ```bash
   python scripts/benchmark_quality.py
   ```

7. **Generate case study report**:
   ```bash
   python scripts/generate_case_study.py
   ```

8. **Validate pruned models** (before upload):
   ```bash
   python scripts/validate_pruned_models.py
   ```

9. **Review results** in `test_data/`:
   - `download_report.json` — Model download status
   - `projections.json/md` — Baseline projections
   - `pruning_report.json/md` — Actual pruning results
   - `pruned_models/` — Directory of pruned model files
   - `comparison_report.json/md` — Projected vs actual comparison
   - `benchmark_report.json/md` — Quality benchmark results
   - `case_study.json/md` — Complete case study with break-even analysis
   - `validation_report.json/md` — Pruned model validation results
   - `validation_results.json` — Post-pruning validation results
   - `test_data/wanda_report.json` — Wanda pruning results
   - `test_data/sparsegpt_report.json` — SparseGPT pruning results
   - `test_data/advanced_pruning_report.json` — Framework comparison results

## Advanced Pruning Frameworks

The repository now includes integration with state-of-the-art pruning frameworks:

### Wanda
Pruning by **W**eights **and** Activations — considers both weight magnitudes and input activation norms for per-output pruning decisions.

### SparseGPT
Layer-wise pruning with reconstruction for GPT models — maintains better quality than magnitude pruning through weight updates.

### Setup
```bash
# Clone the Wanda repo (includes SparseGPT)
git clone --depth 1 https://github.com/locuslab/wanda.git external/wanda
```

### Usage
```bash
# Run Wanda on a model
python scripts/prune_wanda.py --model gpt2 --sparsity 0.3

# Run SparseGPT on a model
python scripts/prune_sparsegpt.py --model gpt2 --sparsity 0.3

# Run batch comparison of both methods
python scripts/prune_advanced_frameworks.py
```

## Actual Results Summary

The pruning exercise revealed significant variance between Atropos projections and actual results
when using unstructured magnitude-based pruning:

- **Savings variance:** -53.4% (actual savings much lower than projected)
- **Only 1 of 8 scenarios** was viable for production deployment
- **OPT models** achieved target sparsity (10-22%), GPT models did not (0.5-7%)
- **Root cause:** Unstructured pruning doesn't reduce memory without sparse tensor support

### Recommendations

1. Use **structured pruning** (LLM-Pruner) for actual memory savings
2. Update Atropos to distinguish between structured vs unstructured pruning projections
3. Test actual pruning methods before making deployment decisions
4. Consider quantization + pruning combinations for better ROI
