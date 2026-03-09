"""Tests for CSV to markdown conversion."""

import tempfile
from pathlib import Path

from atropos.io import csv_to_markdown


def test_csv_to_markdown_basic() -> None:
    """Test converting a simple CSV to markdown."""
    csv_content = (
        "scenario,strategy,memory_gb,"
        "throughput_tok/s,energy_wh_per_request,"
        "annual_savings_usd,break_even_months,"
        "quality_risk,co2e_savings_kg\n"
        "test-scenario,structured_pruning,10.92,"
        "48.00,4.17,12500.50,21.60,medium,525.00\n"
        "test-scenario,mild_pruning,12.60,43.20,"
        "5.21,6240.00,43.27,low,262.50\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

    try:
        markdown = csv_to_markdown(csv_path)

        assert "# Atropos Batch Analysis Report" in markdown
        assert "test-scenario" in markdown
        assert "structured_pruning" in markdown
        assert "|" in markdown  # Has table formatting
        assert "## Aggregate Statistics" in markdown
    finally:
        Path(csv_path).unlink()


def test_csv_to_markdown_output_file() -> None:
    """Test writing markdown to output file."""
    csv_content = """scenario,strategy,annual_savings_usd
scenario-a,strategy-a,10000.00
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "input.csv"
        md_path = Path(tmpdir) / "output.md"

        csv_path.write_text(csv_content)

        result = csv_to_markdown(csv_path, md_path)

        assert md_path.exists()
        assert result == md_path.read_text()
        assert "# Atropos Batch Analysis Report" in md_path.read_text()


def test_csv_to_markdown_empty_file() -> None:
    """Test handling of empty CSV."""
    csv_content = "scenario,strategy,annual_savings_usd\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = f.name

    try:
        markdown = csv_to_markdown(csv_path)
        assert "No data found" in markdown
    finally:
        Path(csv_path).unlink()
