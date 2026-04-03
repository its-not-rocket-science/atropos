from __future__ import annotations

from pathlib import Path


def test_pruning_dockerfiles_exist() -> None:
    expected = [
        Path("docker/pruning_frameworks/wanda/Dockerfile"),
        Path("docker/pruning_frameworks/sparsegpt/Dockerfile"),
        Path("docker/pruning_frameworks/llm-pruner/Dockerfile"),
        Path("docker-compose.yml"),
    ]
    for path in expected:
        assert path.exists(), f"Missing required container config: {path}"
