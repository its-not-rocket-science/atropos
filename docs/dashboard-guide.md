# Atropos Web Dashboard Guide
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.


The Atropos Web Dashboard provides an interactive, browser-based interface for exploring ROI projections for LLM pruning and optimization.

## Launching the Dashboard

### Prerequisites

Install dashboard dependencies:

```bash
pip install dash plotly pandas
```

Or install with all extras:

```bash
pip install -e ".[dashboard]"
```

### Starting the Server

```bash
# Start dashboard on default port (8050)
atropos-llm dashboard

# Start on custom host/port
atropos-llm dashboard --host 0.0.0.0 --port 8080

# Enable debug mode (auto-reload on code changes)
atropos-llm dashboard --debug
```

Then open your browser to `http://localhost:8050` (or the host/port you specified).

## Dashboard Layout

The dashboard has a two-column layout:

- **Left Panel (350px)**: Configuration inputs
- **Right Panel**: Results, charts, and visualizations

### Header Section

The header displays:
- **Atropos** title and logo area
- Subtitle: "ROI Calculator for LLM Pruning & Optimization"

## Configuration Panel

### Scenario Selection

Choose from built-in presets or create custom scenarios:

**Built-in Presets:**
- `edge-coder` - Small model for edge deployment (0.355B params)
- `medium-coder` - Medium-sized coding model (3B params)
- `large-coder` - Large coding deployment (7B params)

**Custom Scenario:**
Select "Custom" to configure your own parameters:
- Model parameters (billions)
- Memory usage (GB)
- Throughput (tokens/sec)
- Power consumption (watts)
- Request volume (per day)
- Tokens per request
- Cost parameters (electricity, hardware, project costs)

### Strategy Selection

Choose optimization strategies:

- `mild_pruning` - Conservative 15% parameter reduction
- `structured_pruning` - 30% structured pruning with speedup
- `hardware_aware_pruning` - Aggressive 50% reduction

**Quantization Bonus:**
Enable the checkbox to combine pruning with quantization for additional gains.

### Region Selection

Select deployment region for accurate carbon calculations:

- **ISO Country Codes**: US, DE, FR, GB, etc.
- **Cloud Regions**: us-east-1, eu-west-1, ap-southeast-1, etc.

Carbon intensity (kg CO2e/kWh) is automatically loaded based on region.

### Action Buttons

- **Calculate** - Run the ROI analysis with selected parameters
- **Export Scenario** - Download the current scenario as YAML

## Results Panel

After clicking Calculate, the results panel displays:

### Summary Cards

Key metrics at a glance:
- **Annual Savings** - Projected cost savings per year
- **Break-even Time** - Months to recover optimization investment
- **CO2e Reduction** - Annual carbon emissions saved
- **Memory Reduction** - Percentage and absolute GB saved
- **Throughput Improvement** - Speedup percentage

### Comparison Chart

Bar chart comparing:
- Baseline vs Optimized memory (GB)
- Baseline vs Optimized throughput (tokens/sec)
- Visual side-by-side comparison

### Savings Breakdown Chart

Pie or stacked bar chart showing:
- Hardware cost savings
- Energy cost savings
- Initial project investment (negative)

### Monte Carlo Uncertainty Chart

When enabled, shows:
- Probability distribution of annual savings
- P5, P25, Median, P75, P95 percentiles
- Risk assessment visualization

## Interactive Features

### Custom Parameter Editing

When "Custom" scenario is selected:

1. Click **Edit Custom Parameters** to open the modal
2. Adjust sliders and inputs:
   - Model size (0.1B to 70B parameters)
   - Memory (1GB to 80GB)
   - Throughput (10 to 1000 tokens/sec)
   - Power (50W to 700W)
   - Request volume (1,000 to 1,000,000/day)
3. Click **Save** to update calculations

### Real-time Updates

All charts update immediately when you click **Calculate**. No page reload required.

### Export Functionality

**Export Scenario:**
- Downloads current configuration as YAML
- Filename: `atropos-scenario-{timestamp}.yaml`
- Can be reloaded via CLI: `atropos-llm scenario file.yaml`

## Advanced Usage

### Monte Carlo Analysis

The dashboard automatically runs Monte Carlo simulations to show uncertainty ranges:

- **Default**: 1000 simulations per calculation
- **Parameters varied**: Memory reduction, throughput improvement
- **Distribution**: Normal with 10% standard deviation

### Strategy Comparison

To compare multiple strategies:

1. Calculate with first strategy
2. Note or screenshot results
3. Change strategy
4. Calculate again
5. Compare side-by-side

### Custom Region Carbon Intensity

If your region isn't listed:

```python
# Add custom preset (for developers)
from atropos.carbon_presets import CARBON_PRESETS

CARBON_PRESETS["CUSTOM"] = CarbonPreset(
    region_code="CUSTOM",
    region_name="My Custom Region",
    carbon_intensity_kg_per_kwh=0.25,
    data_year=2024,
    source="Custom data",
)
```

## Troubleshooting

### Dashboard Won't Start

**Error:** `ModuleNotFoundError: No module named 'dash'`

```bash
pip install dash plotly pandas
```

**Error:** `Address already in use`

```bash
# Use a different port
atropos-llm dashboard --port 8051
```

### Charts Not Displaying

1. Check browser console for JavaScript errors
2. Ensure JavaScript is enabled
3. Try refreshing the page
4. Disable browser extensions that block scripts

### Slow Performance

For faster calculations:
- Disable Monte Carlo (reduce simulations)
- Use preset scenarios instead of custom
- Reduce browser tab count

### Export Not Working

- Check browser download settings
- Ensure pop-ups are not blocked
- Try a different browser

## Configuration Reference

### Default Port

```python
# In your code
from atropos.dashboard import run_dashboard

run_dashboard(host="127.0.0.1", port=8050, debug=False)
```

### Environment Variables

```bash
# Change default port
export ATROPOS_DASHBOARD_PORT=8080

# Enable debug mode
export ATROPOS_DASHBOARD_DEBUG=1
```

## Keyboard Shortcuts

- **Ctrl+Enter** - Trigger Calculate (when focused on input)
- **Esc** - Close modal dialogs

## Browser Compatibility

Tested on:
- Chrome 120+
- Firefox 121+
- Safari 17+
- Edge 120+

Requires JavaScript enabled.

## Next Steps

After using the dashboard:

1. **Export promising scenarios** as YAML
2. **Validate with real models** using `atropos-llm validate`
3. **Run optimization pipeline** using `atropos-llm pipeline`
4. **Share configurations** with your team

## API Integration

The dashboard uses the same core functions as the CLI:

```python
from atropos.calculations import estimate_outcome
from atropos.presets import SCENARIOS, STRATEGIES

scenario = SCENARIOS["medium-coder"]
strategy = STRATEGIES["structured_pruning"]

outcome = estimate_outcome(scenario, strategy)
print(outcome.annual_total_savings_usd)
```

See the [API documentation](api-reference.md) for more details.
