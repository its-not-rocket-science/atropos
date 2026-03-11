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

### 5. discover-models.py (existing)

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

6. **Review results** in `test_data/`:
   - `download_report.json` — Model download status
   - `projections.json` — Baseline projections
   - `projections.md` — Human-readable projection report
   - `pruning_report.json` — Actual pruning results
   - `pruning_results.md` — Human-readable pruning report
   - `pruned_models/` — Directory of pruned model files
   - `comparison_report.json` — Projected vs actual comparison
   - `comparison_report.md` — Human-readable comparison analysis
   - `validation_results.json` — Post-pruning validation results

## Expected Results

- Memory variance: < 20% (Atropos projection vs actual)
- Throughput variance: < 30%
- All models should load successfully on CPU
- GPU recommended for models > 1B parameters
