from __future__ import annotations

from atroposlib.cli.adapters import CliAdapter as ExtractedCliAdapter
from atroposlib.envs.cli_adapter import CliAdapter as LegacyCliAdapter


def test_cli_merge_preserves_yaml_values_not_overridden_by_cli() -> None:
    adapter = ExtractedCliAdapter()

    merged = adapter.merge_yaml_and_cli(
        {"model": "baseline", "workers": 1, "temperature": 0.1},
        {"workers": 4},
    )

    assert merged == {"model": "baseline", "workers": 4, "temperature": 0.1}


def test_legacy_env_cli_adapter_points_to_extracted_adapter() -> None:
    assert LegacyCliAdapter is ExtractedCliAdapter
