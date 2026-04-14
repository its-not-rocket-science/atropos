"""CLI interoperability helpers for environment config adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CliAdapter:
    """Translate environment config maps to CLI args and merge layered config."""

    def build_cli_args(self, config: dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in sorted(config.items()):
            args.extend([f"--{key.replace('_', '-')}", str(value)])
        return args

    def merge_yaml_and_cli(
        self,
        yaml_config: dict[str, Any],
        cli_config: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(yaml_config)
        merged.update(cli_config)
        return merged
