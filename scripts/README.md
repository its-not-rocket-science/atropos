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

### 7. discover-models.py (existing)

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

8. **Review results** in `test_data/`:
   - `download_report.json` — Model download status
   - `projections.json/md` — Baseline projections
   - `pruning_report.json/md` — Actual pruning results
   - `pruned_models/` — Directory of pruned model files
   - `comparison_report.json/md` — Projected vs actual comparison
   - `benchmark_report.json/md` — Quality benchmark results
   - `case_study.json/md` — Complete case study with break-even analysis
   - `validation_results.json` — Post-pruning validation results

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
