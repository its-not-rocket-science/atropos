# Telemetry Collection Guide

This guide shows how to collect real performance metrics from running inference servers and use them to validate Atropos projections.

## Overview

The telemetry collection system allows you to:
1. Connect to running inference servers (vLLM, TGI, Triton)
2. Collect actual performance metrics (memory, throughput, latency)
3. Import metrics into Atropos scenarios
4. Compare projections vs reality

## Prerequisites

```bash
# Install inference server (example: vLLM)
pip install vllm

# Or TGI
# Follow: https://github.com/huggingface/text-generation-inference

# Or Triton
# Follow: https://github.com/triton-inference-server/server
```

## Step 1: Start an Inference Server

### vLLM Example

```bash
# Start vLLM server with a model
python -m vllm.entrypoints.openai.api_server \
    --model gpt2 \
    --port 8000 \
    --max-model-len 2048

# Test it's running
curl http://localhost:8000/health
```

### TGI Example

```bash
# Start TGI server
docker run --gpus all --shm-size 1g -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id gpt2

# Test it's running
curl http://localhost:8080/health
```

### Triton Example

```bash
# Start Triton with your model repository
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/model_repository:/models:rw \
    nvcr.io/nvidia/tritonserver:23.10-py3 \
    tritonserver --model-repository=/models

# Test it's running
curl http://localhost:8000/v2/health/ready
```

## Step 2: Collect Telemetry

### Basic Collection

```bash
# Collect from vLLM for 60 seconds
atropos-llm collect-telemetry \
    --server-type vllm \
    --url http://localhost:8000 \
    --duration 60 \
    --interval 5 \
    --output baseline-telemetry.json

# Output shows collected metrics:
# Collection complete!
# Throughput: 450.3 tok/s
# Latency: 45.2 ms
# Memory: 12.4 GB
```

### Collection with Scenario Creation

```bash
# Collect and automatically create Atropos scenario
atropos-llm collect-telemetry \
    --server-type vllm \
    --url http://localhost:8000 \
    --duration 120 \
    --interval 10 \
    --output baseline-telemetry.json \
    --create-scenario \
    --name "gpt2-baseline"

# Creates both:
# - baseline-telemetry.json (raw telemetry)
# - baseline-telemetry.yaml (Atropos scenario)
```

### Collection Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--server-type` | vllm, tgi, or triton | Required |
| `--url` | Server base URL | http://localhost:8000 |
| `--duration` | Collection duration (seconds) | 60 |
| `--interval` | Sampling interval (seconds) | 5 |
| `--warmup` | Number of warmup requests | 10 |
| `--output` | Output JSON file | Required |
| `--create-scenario` | Also create YAML scenario | False |
| `--name` | Scenario name | {server-type}-collected |

## Step 3: Import Telemetry into Atropos

### Using Collected Telemetry Directly

If you used `--create-scenario`, you already have a YAML file:

```bash
# Use the generated scenario
atropos-llm scenario baseline-telemetry.yaml --strategy structured_pruning
```

### Import from JSON File

```bash
# Import telemetry JSON to create/update scenario
atropos-llm import-telemetry \
    baseline-telemetry.json \
    --format json \
    --name "gpt2-baseline" \
    --output gpt2-scenario.yaml
```

## Step 4: Calibrate Against Projections

### Run Calibration

```bash
# Compare telemetry against Atropos projections
atropos-llm calibrate \
    gpt2-scenario.yaml \
    baseline-telemetry.json \
    --format markdown \
    --output calibration-report.md
```

### Example Calibration Report

```markdown
# Calibration Report: gpt2-baseline

## Variance Summary

| Metric | Projected | Measured | Variance |
|--------|-----------|----------|----------|
| Memory (GB) | 10.5 | 12.4 | +18.1% |
| Throughput (tok/s) | 500 | 450 | -10.0% |
| Latency (ms) | 40 | 45 | +12.5% |

## Assessment

⚠️ 2/3 metrics outside tolerance (>10% variance)

### Recommendations

1. Memory usage higher than expected - update scenario memory_gb
2. Throughput lower - may indicate batch size or hardware differences
3. Consider re-running with actual hardware specs
```

## Complete Workflow Example

### 1. Baseline Measurement

```bash
# Start server with baseline model
python -m vllm.entrypoints.openai.api_server \
    --model gpt2-medium \
    --port 8000 &

# Collect baseline telemetry
atropos-llm collect-telemetry \
    --server-type vllm \
    --url http://localhost:8000 \
    --duration 120 \
    --output baseline.json \
    --create-scenario \
    --name "gpt2-medium-baseline"

# Stop server
pkill -f vllm.entrypoints
```

### 2. Pruned Model Measurement

```bash
# Start server with pruned model (after pruning)
python -m vllm.entrypoints.openai.api_server \
    --model ./pruned-models/gpt2-medium-pruned \
    --port 8000 &

# Collect pruned telemetry
atropos-llm collect-telemetry \
    --server-type vllm \
    --url http://localhost:8000 \
    --duration 120 \
    --output pruned.json \
    --create-scenario \
    --name "gpt2-medium-pruned"
```

### 3. Compare and Validate

```bash
# Create comparison report
cat > compare-telemetry.py << 'EOF'
import json

with open('baseline.json') as f:
    baseline = json.load(f)

with open('pruned.json') as f:
    pruned = json.load(f)

print("## Comparison: Baseline vs Pruned")
print()
print(f"Memory: {baseline['memory_gb']:.1f}GB → {pruned['memory_gb']:.1f}GB "
      f"({(1 - pruned['memory_gb']/baseline['memory_gb'])*100:.1f}% reduction)")
print(f"Throughput: {baseline['throughput_toks_per_sec']:.1f} → "
      f"{pruned['throughput_toks_per_sec']:.1f} tok/s "
      f"({(pruned['throughput_toks_per_sec']/baseline['throughput_toks_per_sec']-1)*100:+.1f}%)")
print(f"Latency: {baseline['latency_ms_per_request']:.1f} → "
      f"{pruned['latency_ms_per_request']:.1f} ms "
      f"({(pruned['latency_ms_per_request']/baseline['latency_ms_per_request']-1)*100:+.1f}%)")
EOF

python compare-telemetry.py
```

## Supported Metrics by Server

### vLLM
- ✅ GPU memory usage (via `/metrics` endpoint)
- ✅ Token throughput (calculated)
- ✅ Request latency
- ✅ Cache hit rates
- ✅ Batch size statistics

### TGI
- ✅ Batch current size
- ✅ Queue size
- ✅ Basic throughput/latency (from benchmark requests)

### Triton
- ✅ Inference statistics
- ✅ Request counts
- ✅ Compute latency
- ⚠️ Memory (requires GPU metrics extension)

## Troubleshooting

### "Server is not healthy"

```bash
# Check server is running
curl http://localhost:8000/health  # vLLM
curl http://localhost:8080/health  # TGI
curl http://localhost:8000/v2/health/ready  # Triton

# Check correct port
# Check firewall/network settings
```

### "No samples collected"

```bash
# Increase duration or decrease interval
atropos-llm collect-telemetry ... --duration 120 --interval 2

# Check server has GPU available
# Check model is loaded and ready
```

### Memory shows 0.0 GB

Some servers don't expose memory metrics directly. The collector falls back to estimates from request benchmarks. For accurate memory:

```bash
# Use nvidia-smi alongside collection
watch -n 5 nvidia-smi

# Or manually specify in scenario
atropos-llm import-telemetry telemetry.json --format json --name "manual" -o scenario.yaml
# Then edit scenario.yaml to set correct memory_gb
```

## Tips for Accurate Collection

1. **Warmup Period**: Allow server to warm up before collecting (handled automatically)
2. **Duration**: Collect for at least 60 seconds to get stable averages
3. **Load**: Ensure server is under realistic load (use actual traffic or load test)
4. **Multiple Runs**: Collect 3 times and average for best accuracy
5. **Document Hardware**: Record GPU type, CPU, RAM for reproducibility

## Next Steps

After collecting telemetry:

1. **Validate Projections**: Use `atropos-llm calibrate` to check accuracy
2. **Update Scenarios**: Adjust scenario parameters based on real data
3. **Run Pipeline**: Use `atropos-llm pipeline` with real scenarios
4. **Share Results**: Export calibration reports to improve Atropos models
