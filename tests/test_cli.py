"""Tests for CLI functionality."""

from pathlib import Path

from atropos.cli import main


def test_list_presets_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Test that list-presets command runs successfully."""
    result = main(["list-presets"])
    captured = capsys.readouterr()
    assert result == 0
    assert "medium-coder" in captured.out
    assert "structured_pruning" in captured.out


def test_preset_json_runs(capsys) -> None:  # type: ignore[no-untyped-def]
    """Test that preset command with JSON report runs successfully."""
    result = main(["preset", "medium-coder", "--report", "json"])
    captured = capsys.readouterr()
    assert result == 0
    assert '"scenario_name": "medium-coder"' in captured.out


def test_compare_markdown_writes_file(tmp_path: Path) -> None:
    """Test that compare command writes markdown output to file."""
    out = tmp_path / "compare.md"
    result = main(
        [
            "compare",
            "medium-coder",
            "--strategies",
            "mild_pruning",
            "structured_pruning",
            "--format",
            "markdown",
            "--output",
            str(out),
        ]
    )
    assert result == 0
    assert out.exists()
    assert "| Strategy |" in out.read_text()
