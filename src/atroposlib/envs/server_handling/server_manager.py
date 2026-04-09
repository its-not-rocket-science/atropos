"""Server launch/environment parsing for local and SLURM execution modes."""

from __future__ import annotations

import os
from dataclasses import dataclass


class ServerManagerError(ValueError):
    """Raised when launch environment variables are missing or malformed."""


@dataclass(frozen=True)
class ServerLaunchConfig:
    """Validated launch configuration derived from process environment."""

    mode: str
    host: str
    port: int
    world_size: int
    rank: int
    local_rank: int
    node_count: int


@dataclass
class ServerManager:
    """Parse and validate cluster/local launch variables for server startup."""

    default_host: str = "127.0.0.1"
    default_port: int = 8000

    def from_environment(self, environ: dict[str, str] | None = None) -> ServerLaunchConfig:
        """Build a validated launch config from environment variables."""

        env = os.environ if environ is None else environ
        mode = self._resolve_mode(env)

        if mode == "localhost":
            return ServerLaunchConfig(
                mode=mode,
                host=env.get("SERVER_HOST", self.default_host),
                port=self._parse_int(
                    env,
                    "SERVER_PORT",
                    default=self.default_port,
                    min_value=1,
                    max_value=65535,
                ),
                world_size=1,
                rank=0,
                local_rank=0,
                node_count=1,
            )

        world_size = self._parse_int(env, "SLURM_NTASKS", min_value=1)
        rank = self._parse_int(env, "SLURM_PROCID", min_value=0)
        local_rank = self._parse_int(env, "SLURM_LOCALID", default=0, min_value=0)
        node_count = self._parse_int(env, "SLURM_JOB_NUM_NODES", default=1, min_value=1)
        host = self._required_str(env, "MASTER_ADDR")
        port = self._parse_int(env, "MASTER_PORT", min_value=1, max_value=65535)

        if rank >= world_size:
            raise ServerManagerError(
                "SLURM_PROCID must be lower than SLURM_NTASKS "
                f"(got rank={rank}, world_size={world_size})."
            )
        if node_count > 1 and host in {"localhost", "127.0.0.1"}:
            raise ServerManagerError(
                "MASTER_ADDR must be a routable hostname/IP for multi-node runs."
            )

        return ServerLaunchConfig(
            mode=mode,
            host=host,
            port=port,
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            node_count=node_count,
        )

    @staticmethod
    def _resolve_mode(env: dict[str, str]) -> str:
        requested_mode = env.get("CLUSTER_LAUNCH_MODE")
        if requested_mode is not None:
            normalized_mode = requested_mode.strip().lower()
            if normalized_mode in {"localhost", "local"}:
                return "localhost"
            if normalized_mode == "slurm":
                return "slurm"
            raise ServerManagerError("CLUSTER_LAUNCH_MODE must be one of: localhost, local, slurm.")

        if any(key.startswith("SLURM_") for key in env):
            return "slurm"

        return "localhost"

    @staticmethod
    def _required_str(env: dict[str, str], key: str) -> str:
        value = env.get(key)
        if value is None or not value.strip():
            raise ServerManagerError(f"Missing required environment variable: {key}")
        return value.strip()

    @staticmethod
    def _parse_int(
        env: dict[str, str],
        key: str,
        *,
        default: int | None = None,
        min_value: int | None = None,
        max_value: int | None = None,
    ) -> int:
        raw = env.get(key)
        if raw is None:
            if default is None:
                raise ServerManagerError(f"Missing required environment variable: {key}")
            return default

        try:
            value = int(raw)
        except ValueError as exc:
            raise ServerManagerError(
                f"Environment variable {key} must be an integer, got {raw!r}."
            ) from exc

        if min_value is not None and value < min_value:
            raise ServerManagerError(
                f"Environment variable {key} must be >= {min_value}, got {value}."
            )
        if max_value is not None and value > max_value:
            raise ServerManagerError(
                f"Environment variable {key} must be <= {max_value}, got {value}."
            )

        return value
