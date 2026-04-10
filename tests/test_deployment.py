"""Tests for deployment automation module."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add src to path for local development
SRC_DIR = Path(__file__).parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from atropos.deployment.health import (  # noqa: E402
    CompositeHealthCheck,
    HealthCheck,
    HttpHealthCheck,
    JsonHealthCheck,
    LatencyHealthCheck,
    get_health_check,
)
from atropos.deployment.models import (  # noqa: E402
    DeploymentRequest,
    DeploymentResult,
    DeploymentStatus,
    DeploymentStrategyType,
)
from atropos.deployment.platforms import (  # noqa: E402
    DeploymentPlatform,
    get_platform,
)
from atropos.deployment.strategies import (  # noqa: E402
    BlueGreenStrategy,
    CanaryStrategy,
    DeploymentStrategy,
    ImmediateStrategy,
    RollingUpdateStrategy,
    get_strategy,
)


# Fixtures
@pytest.fixture
def mock_platform() -> Mock:
    """Mock deployment platform."""
    platform = Mock(spec=DeploymentPlatform)
    platform.deploy.return_value = DeploymentResult(
        request=Mock(spec=DeploymentRequest),
        status=DeploymentStatus.SUCCESS,
        message="Deployed successfully",
        deployment_id="test-deployment-123",
        endpoints=["http://example.com"],
        metrics={"latency_ms": 50.0},
    )
    platform.get_status.return_value = DeploymentResult(
        request=Mock(spec=DeploymentRequest),
        status=DeploymentStatus.SUCCESS,
        message="Healthy",
    )
    platform.rollback.return_value = DeploymentResult(
        request=Mock(spec=DeploymentRequest),
        status=DeploymentStatus.SUCCESS,
        message="Rolled back successfully",
    )
    platform.delete.return_value = DeploymentResult(
        request=Mock(spec=DeploymentRequest),
        status=DeploymentStatus.SUCCESS,
        message="Deleted successfully",
    )
    return platform


@pytest.fixture
def sample_deployment_request() -> DeploymentRequest:
    """Sample deployment request for testing."""
    return DeploymentRequest(
        model_path="/path/to/model",
        platform="vllm",
        strategy=DeploymentStrategyType.IMMEDIATE,
        strategy_config={"initial_percent": 10.0},
        health_checks={"type": "http", "endpoint": "/health"},
        metadata={"environment": "test"},
    )


# Model Tests
class TestDeploymentModels:
    """Tests for deployment models."""

    def test_deployment_request_creation(
        self, sample_deployment_request: DeploymentRequest
    ) -> None:  # noqa: E501
        """Test DeploymentRequest creation and attribute access."""
        assert sample_deployment_request.model_path == "/path/to/model"
        assert sample_deployment_request.platform == "vllm"
        assert sample_deployment_request.strategy == DeploymentStrategyType.IMMEDIATE
        assert sample_deployment_request.strategy_config == {"initial_percent": 10.0}
        assert sample_deployment_request.health_checks == {"type": "http", "endpoint": "/health"}
        assert sample_deployment_request.metadata == {"environment": "test"}

    def test_deployment_result_creation(self) -> None:
        """Test DeploymentResult creation."""
        request = Mock(spec=DeploymentRequest)
        result = DeploymentResult(
            request=request,
            status=DeploymentStatus.SUCCESS,
            message="Test message",
            deployment_id="test-123",
            endpoints=["http://example.com"],
            metrics={"latency": 100},
            start_time="2024-01-01T00:00:00",
            end_time="2024-01-01T00:01:00",
        )
        assert result.request == request
        assert result.status == DeploymentStatus.SUCCESS
        assert result.message == "Test message"
        assert result.deployment_id == "test-123"
        assert result.endpoints == ["http://example.com"]
        assert result.metrics == {"latency": 100}
        assert result.start_time == "2024-01-01T00:00:00"
        assert result.end_time == "2024-01-01T00:01:00"
        assert result.duration_seconds == 60.0

    def test_deployment_result_defaults(self) -> None:
        """Test DeploymentResult with minimal arguments."""
        request = Mock(spec=DeploymentRequest)
        result = DeploymentResult(
            request=request,
            status=DeploymentStatus.FAILED,
            message="Failed",
        )
        assert result.request == request
        assert result.status == DeploymentStatus.FAILED
        assert result.message == "Failed"
        assert result.deployment_id is None
        assert result.endpoints == []
        assert result.metrics == {}
        assert result.start_time is None
        assert result.end_time is None
        assert result.duration_seconds is None

    def test_deployment_status_enum(self) -> None:
        """Test DeploymentStatus enum values."""
        assert str(DeploymentStatus.SUCCESS) == "success"
        assert str(DeploymentStatus.FAILED) == "failed"
        assert str(DeploymentStatus.RUNNING) == "running"

    def test_deployment_strategy_type_enum(self) -> None:
        """Test DeploymentStrategyType enum values."""
        assert str(DeploymentStrategyType.IMMEDIATE) == "immediate"
        assert str(DeploymentStrategyType.CANARY) == "canary"
        assert str(DeploymentStrategyType.BLUE_GREEN) == "blue-green"
        assert str(DeploymentStrategyType.ROLLING) == "rolling"


# Platform Tests
class TestDeploymentPlatforms:
    """Tests for deployment platforms."""

    def test_get_platform(self) -> None:
        """Test get_platform returns correct platform class."""
        # Test with mock platform (actual platforms would need external dependencies)
        with patch.dict("atropos.deployment.platforms.PLATFORMS", {"vllm": Mock}):
            platform = get_platform("vllm", {"config": "value"})
            assert platform is not None

    def test_get_platform_invalid(self) -> None:
        """Test get_platform raises ValueError for invalid platform."""
        with pytest.raises(ValueError, match="Unknown platform"):
            get_platform("invalid-platform", {})

    def test_platform_abstract_methods(self) -> None:
        """Test DeploymentPlatform abstract methods."""
        with pytest.raises(TypeError, match="abstract"):
            _platform = DeploymentPlatform({"config": "value"})

    # Note: Actual platform implementations (VLLMPlatform, TritonPlatform, SageMakerPlatform)
    # would require integration testing with external services. Unit tests would mock
    # external dependencies, but for now we rely on the abstract interface tests.


# Strategy Tests
class TestDeploymentStrategies:
    """Tests for deployment strategies."""

    def test_get_strategy(self) -> None:
        """Test get_strategy returns correct strategy class."""
        strategy = get_strategy("immediate", {})
        assert isinstance(strategy, ImmediateStrategy)

        strategy = get_strategy("canary", {"initial_percent": 5.0})
        assert isinstance(strategy, CanaryStrategy)
        assert strategy.config.get("initial_percent") == 5.0

        strategy = get_strategy("blue-green", {"validation_duration": 30})
        assert isinstance(strategy, BlueGreenStrategy)
        assert strategy.config.get("validation_duration") == 30

        strategy = get_strategy("rolling", {"batch_size": 2})
        assert isinstance(strategy, RollingUpdateStrategy)
        assert strategy.config.get("batch_size") == 2

    def test_get_strategy_invalid(self) -> None:
        """Test get_strategy raises ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("invalid-strategy", {})

    def test_immediate_strategy(
        self, mock_platform: Mock, sample_deployment_request: DeploymentRequest
    ) -> None:  # noqa: E501
        """Test ImmediateStrategy delegates to platform.deploy."""
        strategy = ImmediateStrategy()
        result = strategy.execute(mock_platform, sample_deployment_request)

        mock_platform.deploy.assert_called_once_with(sample_deployment_request)
        assert result == mock_platform.deploy.return_value

    def test_canary_strategy_success(
        self, mock_platform: Mock, sample_deployment_request: DeploymentRequest
    ) -> None:  # noqa: E501
        """Test CanaryStrategy with successful health checks."""
        # Configure mock platform to return success
        mock_platform.get_status.return_value.status = DeploymentStatus.SUCCESS

        strategy = CanaryStrategy(
            {
                "initial_percent": 10.0,
                "increment_percent": 45.0,
                "poll_interval_seconds": 0.0,
                "timeout_seconds": 5.0,
                "max_errors": 3,
            }
        )

        result = strategy.execute(mock_platform, sample_deployment_request)

        assert result.status == DeploymentStatus.SUCCESS
        assert "not managed" in result.message.lower()
        # Should have called deploy once and get_status multiple times
        mock_platform.deploy.assert_called_once()
        assert mock_platform.get_status.call_count == 3

    def test_canary_strategy_failure(
        self, mock_platform: Mock, sample_deployment_request: DeploymentRequest
    ) -> None:  # noqa: E501
        """Test CanaryStrategy with failing health checks triggers rollback."""
        # Configure mock to fail health checks
        mock_platform.get_status.return_value.status = DeploymentStatus.FAILED
        mock_platform.rollback.return_value.status = DeploymentStatus.SUCCESS

        strategy = CanaryStrategy(
            {
                "initial_percent": 10.0,
                "increment_percent": 20.0,
                "poll_interval_seconds": 0.0,
                "timeout_seconds": 5.0,
                "max_errors": 1,  # Fail immediately
            }
        )

        result = strategy.execute(mock_platform, sample_deployment_request)

        assert result.status == DeploymentStatus.FAILED
        assert "rollback" in result.message.lower()
        mock_platform.rollback.assert_called_once()

    def test_blue_green_strategy_success(
        self, mock_platform: Mock, sample_deployment_request: DeploymentRequest
    ) -> None:  # noqa: E501
        """Test BlueGreenStrategy with successful validation."""
        mock_platform.get_status.return_value.status = DeploymentStatus.SUCCESS

        strategy = BlueGreenStrategy(
            {
                "timeout_seconds": 0.01,
                "poll_interval_seconds": 0.0,
                "auto_swap": True,
            }
        )

        result = strategy.execute(mock_platform, sample_deployment_request)

        assert result.status == DeploymentStatus.SUCCESS
        assert "blue-green" in result.message.lower()
        mock_platform.deploy.assert_called_once()
        mock_platform.get_status.assert_called()
        assert sample_deployment_request.metadata == {"environment": "test"}

    def test_blue_green_strategy_failure(
        self, mock_platform: Mock, sample_deployment_request: DeploymentRequest
    ) -> None:  # noqa: E501
        """Test BlueGreenStrategy with failed validation."""
        mock_platform.get_status.return_value.status = DeploymentStatus.FAILED

        strategy = BlueGreenStrategy(
            {
                "timeout_seconds": 0.01,
                "poll_interval_seconds": 0.0,
                "auto_swap": True,
            }
        )

        result = strategy.execute(mock_platform, sample_deployment_request)

        assert result.status == DeploymentStatus.FAILED
        assert "cleaned up" in result.message.lower()
        mock_platform.delete.assert_called_once()

    def test_blue_green_validation_failure_cleanup(
        self, mock_platform: Mock, sample_deployment_request: DeploymentRequest
    ) -> None:
        """Validation failures should clean up green deployment and preserve request."""
        mock_platform.get_status.return_value.status = DeploymentStatus.FAILED
        mock_platform.get_status.return_value.message = "unhealthy"
        strategy = BlueGreenStrategy({"timeout_seconds": 1.0, "poll_interval_seconds": 0.0})

        _ = strategy.execute(mock_platform, sample_deployment_request)

        deployed_request = mock_platform.deploy.call_args.args[0]
        assert deployed_request.metadata["environment"] == "green"
        assert sample_deployment_request.metadata == {"environment": "test"}
        mock_platform.delete.assert_called_once_with("test-deployment-123")

    def test_strategy_config_parsing(self) -> None:
        """Strategy config should parse time options and compatibility defaults."""
        canary_default = CanaryStrategy({"interval_minutes": 2})
        assert canary_default.poll_interval_seconds == 120.0
        assert canary_default.timeout_seconds == 300.0

        canary_override = CanaryStrategy({"poll_interval_seconds": 3, "timeout_seconds": 12})
        assert canary_override.poll_interval_seconds == 3.0
        assert canary_override.timeout_seconds == 12.0

        blue_green = BlueGreenStrategy({"validation_duration": 2})
        assert blue_green.timeout_seconds == 120.0

    def test_rolling_update_strategy(
        self, mock_platform: Mock, sample_deployment_request: DeploymentRequest
    ) -> None:  # noqa: E501
        """Test RollingUpdateStrategy delegates to platform.deploy."""
        strategy = RollingUpdateStrategy()
        result = strategy.execute(mock_platform, sample_deployment_request)

        mock_platform.deploy.assert_called_once_with(sample_deployment_request)
        assert result.status == mock_platform.deploy.return_value.status
        assert "not managed" in result.message.lower()

    def test_strategy_abstract_methods(self) -> None:
        """Test DeploymentStrategy abstract methods."""
        with pytest.raises(TypeError, match="abstract"):
            _strategy = DeploymentStrategy({"config": "value"})


# Health Check Tests
class TestHealthChecks:
    """Tests for health checks."""

    def test_get_health_check(self) -> None:
        """Test get_health_check returns correct health check class."""
        check = get_health_check("http", {})
        assert isinstance(check, HttpHealthCheck)

        check = get_health_check("latency", {"max_latency_ms": 500})
        assert isinstance(check, LatencyHealthCheck)
        assert check.config.get("max_latency_ms") == 500

        check = get_health_check("json", {"expected_fields": {"status": "ok"}})
        assert isinstance(check, JsonHealthCheck)
        assert check.config.get("expected_fields") == {"status": "ok"}

        check = get_health_check("composite", {"checks": []})
        assert isinstance(check, CompositeHealthCheck)
        assert check.config.get("checks") == []

    def test_get_health_check_invalid(self) -> None:
        """Test get_health_check raises ValueError for invalid type."""
        with pytest.raises(ValueError, match="Unknown health check type"):
            get_health_check("invalid-type", {})

    def test_http_health_check_success(self) -> None:
        """Test HttpHealthCheck with successful response."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.getcode.return_value = 200
            mock_response.read.return_value = b"OK"
            mock_urlopen.return_value.__enter__.return_value = mock_response

            check = HttpHealthCheck({"expected_status": 200, "expected_text": "OK"})
            result = check.check("http://example.com/health")

            assert result["healthy"] is True
            assert "latency_ms" in result
            assert result["status_code"] == 200

    def test_http_health_check_failure(self) -> None:
        """Test HttpHealthCheck with failed response."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.getcode.return_value = 500
            mock_response.read.return_value = b"Error"
            mock_urlopen.return_value.__enter__.return_value = mock_response

            check = HttpHealthCheck({"expected_status": 200})
            result = check.check("http://example.com/health")

            assert result["healthy"] is False
            assert "error" in result

    def test_latency_health_check(self) -> None:
        """Test LatencyHealthCheck latency threshold."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.getcode.return_value = 200
            mock_response.read.return_value = b"OK"
            mock_urlopen.return_value.__enter__.return_value = mock_response

            check = LatencyHealthCheck({"max_latency_ms": 100})
            result = check.check("http://example.com/health")

            # Mock latency will be very small
            assert result["healthy"] is True
            assert "latency_healthy" in result
            assert result["latency_healthy"] is True

    def test_json_health_check_success(self) -> None:
        """Test JsonHealthCheck with valid JSON."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.getcode.return_value = 200
            mock_response.read.return_value = b'{"status": "ok", "version": "1.0"}'
            mock_urlopen.return_value.__enter__.return_value = mock_response

            check = JsonHealthCheck({"expected_fields": {"status": "ok"}})
            result = check.check("http://example.com/health")

            assert result["healthy"] is True
            assert "json_data" in result
            assert result["json_data"]["status"] == "ok"

    def test_json_health_check_failure(self) -> None:
        """Test JsonHealthCheck with invalid JSON or field mismatch."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.getcode.return_value = 200
            mock_response.read.return_value = b'{"status": "error"}'
            mock_urlopen.return_value.__enter__.return_value = mock_response

            check = JsonHealthCheck({"expected_fields": {"status": "ok"}})
            result = check.check("http://example.com/health")

            assert result["healthy"] is False
            assert "error" in result

    def test_composite_health_check_all(self) -> None:
        """Test CompositeHealthCheck with require_all=True."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = Mock()
            mock_response.getcode.return_value = 200
            mock_response.read.return_value = b"OK"
            mock_urlopen.return_value.__enter__.return_value = mock_response

            check = CompositeHealthCheck(
                {
                    "checks": [
                        {"type": "http", "expected_status": 200},
                        {"type": "latency", "max_latency_ms": 1000},
                    ],
                    "require_all": True,
                }
            )
            result = check.check("http://example.com/health")

            assert result["healthy"] is True
            assert "checks" in result
            assert len(result["checks"]) == 2

    def test_composite_health_check_any(self) -> None:
        """Test CompositeHealthCheck with require_all=False."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            # Create two mock responses
            mock_response1 = Mock()
            mock_response1.getcode.return_value = 500
            mock_response1.read.return_value = b""
            mock_response2 = Mock()
            mock_response2.getcode.return_value = 200
            mock_response2.read.return_value = b""

            # Create two mock context managers using MagicMock
            mock_context1 = MagicMock()
            mock_context1.__enter__.return_value = mock_response1
            mock_context1.__exit__.return_value = None
            mock_context2 = MagicMock()
            mock_context2.__enter__.return_value = mock_response2
            mock_context2.__exit__.return_value = None

            # Set side effect to return each context manager in order
            mock_urlopen.side_effect = [mock_context1, mock_context2]

            check = CompositeHealthCheck(
                {
                    "checks": [
                        {"type": "http", "expected_status": 200},
                        {"type": "http", "expected_status": 200},
                    ],
                    "require_all": False,
                }
            )
            result = check.check("http://example.com/health")

            # Should be healthy because at least one check passes
            assert result["healthy"] is True

    def test_health_check_abstract_methods(self) -> None:
        """Test HealthCheck abstract methods."""
        with pytest.raises(TypeError, match="abstract"):
            _check = HealthCheck({"config": "value"})
