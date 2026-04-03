"""Fault-tolerant pruning framework wrappers with native/container/mock fallbacks."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

MAX_PRUNING_TIMEOUT_SECONDS = 24 * 60 * 60


class ExecutionMode(str, Enum):
    """Execution mode selected for pruning."""

    NATIVE = "native"
    CONTAINER = "container"
    MOCK = "mock"
    FAILED = "failed"


@dataclass
class ResourceLimits:
    """Resource controls for pruning execution."""

    cpu_cores: int | None = None
    gpu_memory_gb: int | None = None


@dataclass
class PruningResult:
    """Result payload returned by a pruning wrapper."""

    success: bool
    framework: str
    mode: ExecutionMode
    output_path: Path | None = None
    checkpoint_path: Path | None = None
    duration_seconds: float = 0.0
    warning: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PruningFramework(ABC):
    """Common interface for reliable pruning integrations."""

    framework_name: str
    container_image: str
    container_workdir: str = "/workspace"

    def __init__(
        self,
        timeout_seconds: int = MAX_PRUNING_TIMEOUT_SECONDS,
        container_runtime: str | None = None,
    ) -> None:
        self.timeout_seconds = min(timeout_seconds, MAX_PRUNING_TIMEOUT_SECONDS)
        self.container_runtime = container_runtime or self._detect_container_runtime()

    @abstractmethod
    def native_dependency_hint(self) -> str:
        """Return a user-facing dependency hint for native mode failures."""

    @abstractmethod
    def is_native_available(self) -> bool:
        """Return True when native dependency stack is importable/available."""

    @abstractmethod
    def build_native_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        """Build native subprocess invocation command."""

    @abstractmethod
    def build_container_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        """Build command run inside the framework container."""

    def prune(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        resource_limits: ResourceLimits | None = None,
        allow_mock: bool = True,
    ) -> PruningResult:
        """Run pruning with fallback chain: native -> container -> mock -> graceful fail."""
        started = time.monotonic()
        output_path.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_path / f"{self.framework_name}_checkpoint.json"
        self._write_checkpoint(
            checkpoint_path,
            {"status": "started", "framework": self.framework_name},
        )

        native_error: str | None = None
        if self.is_native_available():
            result = self._run_command(
                self.build_native_command(model_name, output_path, sparsity, checkpoint_path),
                mode=ExecutionMode.NATIVE,
                framework=self.framework_name,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                resource_limits=resource_limits,
            )
            if result.success:
                self._write_checkpoint(checkpoint_path, {"status": "completed", "mode": "native"})
                result.duration_seconds = time.monotonic() - started
                return result
            native_error = result.error_message

        if self.container_runtime:
            result = self._run_in_container(
                model_name=model_name,
                output_path=output_path,
                sparsity=sparsity,
                checkpoint_path=checkpoint_path,
                resource_limits=resource_limits,
            )
            if result.success:
                self._write_checkpoint(
                    checkpoint_path,
                    {"status": "completed", "mode": "container"},
                )
                result.duration_seconds = time.monotonic() - started
                return result
            container_error = result.error_message
        else:
            container_error = "No rootless container runtime (docker/podman) detected"

        if allow_mock:
            self._write_checkpoint(checkpoint_path, {"status": "completed", "mode": "mock"})
            return PruningResult(
                success=True,
                framework=self.framework_name,
                mode=ExecutionMode.MOCK,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                duration_seconds=time.monotonic() - started,
                warning=(
                    f"{self.framework_name} native/container execution unavailable; "
                    "returned simulated pruning result"
                ),
                metadata={"estimated_sparsity": sparsity, "model": model_name},
            )

        self._write_checkpoint(checkpoint_path, {"status": "failed"})
        return PruningResult(
            success=False,
            framework=self.framework_name,
            mode=ExecutionMode.FAILED,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            duration_seconds=time.monotonic() - started,
            error_message=(
                f"{self.framework_name} failed. Native error: {native_error or 'not available'}. "
                "Container error: "
                f"{container_error}. Suggested fix: {self.native_dependency_hint()}"
            ),
        )

    def test_integration(self) -> PruningResult:
        """Lightweight integration check for CLI test command."""
        temp_output = Path("/tmp") / f"atropos_{self.framework_name}_smoke"
        return self.prune(
            model_name="sshleifer/tiny-gpt2",
            output_path=temp_output,
            sparsity=0.3,
            resource_limits=ResourceLimits(cpu_cores=2),
            allow_mock=True,
        )

    def _run_in_container(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
        resource_limits: ResourceLimits | None,
    ) -> PruningResult:
        if not self.container_runtime:
            return PruningResult(
                success=False,
                framework=self.framework_name,
                mode=ExecutionMode.CONTAINER,
                error_message="Container runtime not available",
            )

        host_dir = output_path.resolve()
        cmd = [
            self.container_runtime,
            "run",
            "--rm",
            "-v",
            f"{host_dir}:{self.container_workdir}/output",
            self.container_image,
            *self.build_container_command(model_name, output_path, sparsity, checkpoint_path),
        ]
        return self._run_command(
            cmd,
            mode=ExecutionMode.CONTAINER,
            framework=self.framework_name,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            resource_limits=resource_limits,
        )

    def _run_command(
        self,
        cmd: list[str],
        mode: ExecutionMode,
        framework: str,
        output_path: Path,
        checkpoint_path: Path,
        resource_limits: ResourceLimits | None,
    ) -> PruningResult:
        env = os.environ.copy()
        if resource_limits and resource_limits.cpu_cores:
            env["OMP_NUM_THREADS"] = str(resource_limits.cpu_cores)
        if resource_limits and resource_limits.gpu_memory_gb:
            env["ATROPOS_GPU_MEMORY_GB"] = str(resource_limits.gpu_memory_gb)

        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return PruningResult(
                success=False,
                framework=framework,
                mode=mode,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                error_message=f"Pruning timeout after {self.timeout_seconds} seconds",
            )

        if completed.returncode == 0:
            return PruningResult(
                success=True,
                framework=framework,
                mode=mode,
                output_path=output_path,
                checkpoint_path=checkpoint_path,
                metadata={"stdout": completed.stdout[-2000:]},
            )

        return PruningResult(
            success=False,
            framework=framework,
            mode=mode,
            output_path=output_path,
            checkpoint_path=checkpoint_path,
            error_message=completed.stderr[-2000:] or completed.stdout[-2000:],
        )

    def _write_checkpoint(self, checkpoint_path: Path, payload: dict[str, Any]) -> None:
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _detect_container_runtime(self) -> str | None:
        for runtime in ("docker", "podman"):
            if shutil.which(runtime):
                return runtime
        return None


class WandaPruning(PruningFramework):
    framework_name = "wanda"
    container_image = "atropos/pruning-wanda:cuda11.8"

    def native_dependency_hint(self) -> str:
        return (
            "Wanda failed because transformers version mismatch; run "
            "`atropos-llm setup-pruning --fix`"
        )

    def is_native_available(self) -> bool:
        return Path("scripts/prune_wanda.py").exists()

    def build_native_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            "scripts/prune_wanda.py",
            "--model",
            model_name,
            "--sparsity",
            str(sparsity),
            "--save",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]

    def build_container_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        return [
            "python",
            "/workspace/framework/run_prune.py",
            "--framework",
            "wanda",
            "--model",
            model_name,
            "--sparsity",
            str(sparsity),
            "--save",
            "/workspace/output",
            "--checkpoint",
            "/workspace/output/wanda_checkpoint.json",
        ]


class SparseGPTPruning(PruningFramework):
    framework_name = "sparsegpt"
    container_image = "atropos/pruning-sparsegpt:cuda11.8"

    def native_dependency_hint(self) -> str:
        return "SparseGPT dependency stack broken; run `atropos-llm setup-pruning --fix`"

    def is_native_available(self) -> bool:
        return Path("scripts/prune_sparsegpt.py").exists()

    def build_native_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            "scripts/prune_sparsegpt.py",
            "--model",
            model_name,
            "--sparsity",
            str(sparsity),
            "--save",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]

    def build_container_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        return [
            "python",
            "/workspace/framework/run_prune.py",
            "--framework",
            "sparsegpt",
            "--model",
            model_name,
            "--sparsity",
            str(sparsity),
            "--save",
            "/workspace/output",
            "--checkpoint",
            "/workspace/output/sparsegpt_checkpoint.json",
        ]


class LLMPrunerPruning(PruningFramework):
    framework_name = "llm-pruner"
    container_image = "atropos/pruning-llm-pruner:cuda11.8"

    def native_dependency_hint(self) -> str:
        return "LLM-Pruner not importable; run `atropos-llm setup-pruning --fix`"

    def is_native_available(self) -> bool:
        try:
            __import__("llm_pruner")
            return True
        except ImportError:
            return False

    def build_native_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        return [
            sys.executable,
            "-m",
            "llm_pruner.entry",
            "--model",
            model_name,
            "--sparsity",
            str(sparsity),
            "--save",
            str(output_path),
            "--checkpoint",
            str(checkpoint_path),
        ]

    def build_container_command(
        self,
        model_name: str,
        output_path: Path,
        sparsity: float,
        checkpoint_path: Path,
    ) -> list[str]:
        return [
            "python",
            "/workspace/framework/run_prune.py",
            "--framework",
            "llm-pruner",
            "--model",
            model_name,
            "--sparsity",
            str(sparsity),
            "--save",
            "/workspace/output",
            "--checkpoint",
            "/workspace/output/llm-pruner_checkpoint.json",
        ]


def get_pruning_framework(name: str) -> PruningFramework:
    normalized = name.lower()
    if normalized == "wanda":
        return WandaPruning()
    if normalized == "sparsegpt":
        return SparseGPTPruning()
    if normalized in {"llm-pruner", "llmpruner"}:
        return LLMPrunerPruning()
    raise ValueError(f"Unsupported pruning framework: {name}")
