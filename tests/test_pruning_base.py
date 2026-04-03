from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from atropos.pruning.base import ExecutionMode, WandaPruning, get_pruning_framework


def test_get_pruning_framework() -> None:
    framework = get_pruning_framework("wanda")
    assert framework.framework_name == "wanda"


@patch("atropos.pruning.base.subprocess.run")
def test_native_success(mock_run: MagicMock, tmp_path: Path) -> None:
    completed = MagicMock()
    completed.returncode = 0
    completed.stdout = "ok"
    completed.stderr = ""
    mock_run.return_value = completed

    framework = WandaPruning(container_runtime="docker")
    result = framework.prune("sshleifer/tiny-gpt2", tmp_path / "out", 0.3)

    assert result.success is True
    assert result.mode == ExecutionMode.NATIVE


@patch("atropos.pruning.base.subprocess.run")
def test_container_fallback(mock_run: MagicMock, tmp_path: Path) -> None:
    native_fail = MagicMock(returncode=1, stdout="", stderr="native failed")
    container_ok = MagicMock(returncode=0, stdout="container ok", stderr="")
    mock_run.side_effect = [native_fail, container_ok]

    framework = WandaPruning(container_runtime="docker")
    result = framework.prune("sshleifer/tiny-gpt2", tmp_path / "out", 0.3)

    assert result.success is True
    assert result.mode == ExecutionMode.CONTAINER


@patch("atropos.pruning.base.subprocess.run")
def test_mock_fallback(mock_run: MagicMock, tmp_path: Path) -> None:
    native_fail = MagicMock(returncode=1, stdout="", stderr="native failed")
    container_fail = MagicMock(returncode=1, stdout="", stderr="container failed")
    mock_run.side_effect = [native_fail, container_fail]

    framework = WandaPruning(container_runtime="docker")
    result = framework.prune("sshleifer/tiny-gpt2", tmp_path / "out", 0.3, allow_mock=True)

    assert result.success is True
    assert result.mode == ExecutionMode.MOCK
    assert result.warning is not None
