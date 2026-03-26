"""Unit tests for distributed pruning functionality."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from atropos.pipeline.config import PipelineConfig, PruningConfig
from atropos.pruning_integration import (
    PRUNING_FRAMEWORKS,
    DistributedPruningWrapper,
    PruningResult,
    get_pruning_framework,
)


class MockPruningFramework:
    """Mock pruning framework for testing."""

    def __init__(self, config=None):
        self.config = config
        self.prune_called = False
        self.prune_kwargs = None

    def _check_availability(self) -> bool:
        return True

    def prune(
        self,
        model_name: str,
        output_path: Path,
        target_sparsity: float,
        **kwargs,
    ) -> PruningResult:
        self.prune_called = True
        self.prune_kwargs = kwargs
        return PruningResult(
            success=True,
            original_params=1000,
            pruned_params=700,
            sparsity_achieved=0.3,
        )

    def estimate_pruning_time(self, model_params_b: float) -> float:
        return model_params_b * 0.1  # dummy scaling


def test_distributed_pruning_wrapper_initialization():
    """Test DistributedPruningWrapper initialization."""
    mock_framework = MockPruningFramework()
    config = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            framework="wanda-patched",
            distributed=True,
            num_gpus=2,
            parallel_strategy="data",
        ),
    )

    wrapper = DistributedPruningWrapper(mock_framework, config)

    assert wrapper.wrapped_framework == mock_framework
    assert wrapper.distributed_config.distributed is True
    assert wrapper.distributed_config.num_gpus == 2
    assert wrapper.distributed_config.parallel_strategy == "data"


@patch("atropos.pruning_integration.distributed_context")
def test_distributed_pruning_wrapper_prune_with_distributed(
    mock_distributed_context,
):
    """Test wrapper prune method with distributed enabled."""
    # Mock distributed context
    mock_ctx = Mock()
    mock_ctx.rank = 1
    mock_ctx.world_size = 2
    mock_ctx.local_rank = 0
    mock_distributed_context.return_value.__enter__.return_value = mock_ctx

    mock_framework = MockPruningFramework()
    config = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            framework="wanda-patched",
            distributed=True,
            num_gpus=2,
        ),
    )

    wrapper = DistributedPruningWrapper(mock_framework, config)
    result = wrapper.prune("gpt2", Path("/tmp/output"), 0.3, extra_arg="value")

    # Verify distributed context was used
    mock_distributed_context.assert_called_once_with(wrapper.distributed_config)

    # Verify wrapped framework was called with distributed kwargs
    assert mock_framework.prune_called is True
    assert mock_framework.prune_kwargs["rank"] == 1
    assert mock_framework.prune_kwargs["world_size"] == 2
    assert mock_framework.prune_kwargs["local_rank"] == 0
    assert mock_framework.prune_kwargs["extra_arg"] == "value"
    assert "distributed_config" in mock_framework.prune_kwargs

    # Verify result
    assert result.success is True
    assert result.sparsity_achieved == 0.3


def test_distributed_pruning_wrapper_prune_without_distributed():
    """Test wrapper prune method when distributed is disabled."""
    mock_framework = MockPruningFramework()
    config = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            framework="wanda-patched",
            distributed=False,  # Disabled
        ),
    )

    wrapper = DistributedPruningWrapper(mock_framework, config)
    result = wrapper.prune("gpt2", Path("/tmp/output"), 0.3)

    # Should call wrapped framework directly without distributed context
    assert mock_framework.prune_called is True
    assert mock_framework.prune_kwargs == {}  # No distributed kwargs added
    assert result.success is True


def test_distributed_pruning_wrapper_estimate_pruning_time():
    """Test time estimation with scaling factors."""
    mock_framework = MockPruningFramework()

    # Test with distributed disabled
    config_no_dist = PipelineConfig(
        name="test",
        pruning=PruningConfig(distributed=False, num_gpus=1),
    )
    wrapper_no_dist = DistributedPruningWrapper(mock_framework, config_no_dist)
    base_time = mock_framework.estimate_pruning_time(100.0)
    estimated_no_dist = wrapper_no_dist.estimate_pruning_time(100.0)
    assert estimated_no_dist == base_time  # No scaling

    # Test with data parallelism
    config_data = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            distributed=True,
            num_gpus=4,
            parallel_strategy="data",
        ),
    )
    wrapper_data = DistributedPruningWrapper(mock_framework, config_data)
    estimated_data = wrapper_data.estimate_pruning_time(100.0)
    # Should be base_time / (num_gpus * scaling_factor)
    expected_data = base_time / (4 * 0.8)  # 80% efficiency
    assert estimated_data == pytest.approx(expected_data)

    # Test with layer parallelism
    config_layer = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            distributed=True,
            num_gpus=4,
            parallel_strategy="layer",
        ),
    )
    wrapper_layer = DistributedPruningWrapper(mock_framework, config_layer)
    estimated_layer = wrapper_layer.estimate_pruning_time(100.0)
    expected_layer = base_time / (4 * 0.6)  # 60% efficiency
    assert estimated_layer == pytest.approx(expected_layer)

    # Test with model parallelism
    config_model = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            distributed=True,
            num_gpus=4,
            parallel_strategy="model",
        ),
    )
    wrapper_model = DistributedPruningWrapper(mock_framework, config_model)
    estimated_model = wrapper_model.estimate_pruning_time(100.0)
    expected_model = base_time / (4 * 0.4)  # 40% efficiency
    assert estimated_model == pytest.approx(expected_model)


def test_get_pruning_framework_with_distributed():
    """Test that get_pruning_framework wraps frameworks when distributed=True."""
    config = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            framework="wanda-patched",
            distributed=True,
            num_gpus=2,
        ),
    )

    # Create a mock framework class
    mock_framework_class = Mock()
    mock_framework = Mock()
    mock_framework_class.return_value = mock_framework

    # Temporarily replace the entry in PRUNING_FRAMEWORKS
    original_class = PRUNING_FRAMEWORKS["wanda-patched"]
    try:
        PRUNING_FRAMEWORKS["wanda-patched"] = mock_framework_class
        framework = get_pruning_framework("wanda-patched", config)
    finally:
        PRUNING_FRAMEWORKS["wanda-patched"] = original_class

    # Should return a DistributedPruningWrapper
    assert isinstance(framework, DistributedPruningWrapper)
    assert framework.wrapped_framework == mock_framework
    mock_framework_class.assert_called_once_with(config)


def test_get_pruning_framework_without_distributed():
    """Test that get_pruning_framework returns base framework when distributed=False."""
    config = PipelineConfig(
        name="test",
        pruning=PruningConfig(
            framework="wanda-patched",
            distributed=False,
        ),
    )

    # Create a mock framework class
    mock_framework_class = Mock()
    mock_framework = Mock()
    mock_framework_class.return_value = mock_framework

    # Temporarily replace the entry in PRUNING_FRAMEWORKS
    original_class = PRUNING_FRAMEWORKS["wanda-patched"]
    try:
        PRUNING_FRAMEWORKS["wanda-patched"] = mock_framework_class
        framework = get_pruning_framework("wanda-patched", config)
    finally:
        PRUNING_FRAMEWORKS["wanda-patched"] = original_class

    # Should return the base framework directly (no wrapper)
    assert framework == mock_framework
    mock_framework_class.assert_called_once_with(config)


def test_get_pruning_framework_no_config():
    """Test get_pruning_framework with None config."""
    # Create a mock framework class
    mock_framework_class = Mock()
    mock_framework = Mock()
    mock_framework_class.return_value = mock_framework

    # Temporarily replace the entry in PRUNING_FRAMEWORKS
    original_class = PRUNING_FRAMEWORKS["wanda-patched"]
    try:
        PRUNING_FRAMEWORKS["wanda-patched"] = mock_framework_class
        framework = get_pruning_framework("wanda-patched", None)
    finally:
        PRUNING_FRAMEWORKS["wanda-patched"] = original_class

    assert framework == mock_framework
    mock_framework_class.assert_called_once_with(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
