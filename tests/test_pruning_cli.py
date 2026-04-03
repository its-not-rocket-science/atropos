from __future__ import annotations

from unittest.mock import patch

from atropos.cli import main
from atropos.pruning.base import ExecutionMode, PruningResult
from atropos.pruning.manager import SetupSummary


@patch("atropos.cli.setup_pruning_environment")
def test_setup_pruning_command(mock_setup) -> None:  # type: ignore[no-untyped-def]
    mock_setup.return_value = SetupSummary(success=True, message="ok")
    assert main(["setup-pruning"]) == 0


@patch("atropos.cli.test_pruning_framework")
def test_test_pruning_command(mock_test) -> None:  # type: ignore[no-untyped-def]
    mock_test.return_value = PruningResult(
        success=True,
        framework="wanda",
        mode=ExecutionMode.NATIVE,
    )
    assert main(["test-pruning", "--framework", "wanda"]) == 0
