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

### 2. test_pruning_candidates.py

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

### 3. discover-models.py (existing)

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

2. **Run validation tests**:
   ```bash
   python scripts/test_pruning_candidates.py
   ```

3. **Review results** in `test_data/download_report.json` and `test_data/validation_results.json`

## Expected Results

- Memory variance: < 20% (Atropos projection vs actual)
- Throughput variance: < 30%
- All models should load successfully on CPU
- GPU recommended for models > 1B parameters
