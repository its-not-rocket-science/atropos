# Cloud Cost Modeling Methodology

Atropos now supports cloud-first deployment economics with offline pricing caches and optional live refresh.

## What is modeled

- **On-demand instances** (`ondemand`) with hourly rates.
- **Spot/Preemptible** (`spot`) with hourly rates + interruption probability warnings.
- **Reserved commitments** (`reserved`) with 1-year and 3-year discount schedules plus commitment buyout exposure.
- **Serverless inference** using per-inference + per-second + GB-second terms.
- **Ancillary spend** split into compute, storage, and data-transfer egress.

## Supported providers

- AWS (EC2 GPU classes + SageMaker references + Lambda serverless baseline)
- Azure (NCasT4_v3, NC A100 v4, AML endpoint references)
- GCP (G2 L4, A2 A100, Vertex AI references)
- GPU rental catalogs (Lambda Labs, RunPod, Vast.ai, Together.ai)
- Serverless providers (Replicate, Banana, Modal)

## Data source strategy

1. **Offline-first cache** in `data/cloud_pricing_YYYY-MM-DD.json`.
2. At runtime, Atropos uses the most recent cache file less than 30 days old.
3. If no valid cache is found, Atropos falls back to built-in static provider catalogs.
4. Optional `--fetch-live-pricing` triggers a refresh hook and writes new cache data.

## Scenario schema additions

```yaml
deployment:
  platform: aws
  instance_type: p4d.24xlarge
  purchase_option: spot  # ondemand | spot | reserved
  region: us-east-1
  commitment_years: 1

monthly_runtime_hours: 730
monthly_inference_count: 120000
average_duration_seconds: 1.2
average_memory_gb: 16
monthly_storage_gb: 200
monthly_data_transfer_gb: 600
currency: USD
```

## CLI workflows

- `atropos-llm cloud-pricing list-providers`
- `atropos-llm cloud-pricing estimate --scenario scenario.yaml --provider aws`
- `atropos-llm cloud-pricing compare --scenario scenario.yaml --providers aws,azure,lambda-labs`

## Notes on transparency and roadmap

- Every estimate includes a pricing-source timestamp from the loaded catalog.
- Currency conversion uses cache-provided daily rates with USD as base.
- CI should mock live APIs; production updates should be captured by weekly cache refresh.
- Enterprise integrations (e.g., Infracost/Vantage) can plug into the same catalog layer.
