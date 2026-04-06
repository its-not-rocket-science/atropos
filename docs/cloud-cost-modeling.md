# Cloud Cost Modeling Methodology

Atropos now supports cloud-native pricing for AWS, Azure, GCP, GPU rental providers, and serverless inference providers.

## Data sources

- Offline cache: `data/cloud_pricing_YYYY-MM-DD.json` (required for basic operation).
- Optional live refresh: `scripts/update_cloud_pricing.py` and CLI `--fetch-live-pricing`.
- Pricing catalogs include timestamp (`as_of_date`) and should be refreshed weekly.

## Cost model types

1. **On-demand**: fixed `USD/hour` multiplied by runtime hours.
2. **Spot/Preemptible**: discounted `USD/hour` with interruption probability; Atropos applies expected retry overhead and emits a risk warning.
3. **Reserved/Savings Plans**: 1-year or 3-year commitment discount. Buyout exposure is included as a separate estimate field.
4. **Serverless**: `USD/inference + USD/GB-second`.

## ROI decomposition

Each estimate is broken into:

- Compute cost
- Storage cost
- Network (egress) cost

This enables pruning-sensitive scenarios (smaller model artifacts) to reflect reduced transfer and storage impacts.

## Granularity

Estimates support hourly, monthly, and annual views.

## Currency support

- USD is the base currency.
- Offline catalog includes daily FX rates for local-currency display where available.

## Commands

- `atropos-llm cloud-pricing list-providers`
- `atropos-llm cloud-pricing estimate --scenario scenario.yaml --provider aws`
- `atropos-llm cloud-pricing compare --scenario scenario.yaml --providers aws,azure,lambda-labs`

