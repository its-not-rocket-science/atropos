"""Seed management utilities for reproducible Atropos runs."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


def _normalize_seed(seed: int) -> int:
    if seed < 0:
        raise ValueError("Seed must be non-negative.")
    return seed % (2**31)


@dataclass(frozen=True, slots=True)
class SeedManager:
    """Derive deterministic, component-scoped seeds from a single root seed."""

    root_seed: int
    derivation_version: str = "v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "root_seed", _normalize_seed(self.root_seed))

    def derive_seed(
        self,
        component: str,
        *,
        stage: str = "default",
        rank: int = 0,
        worker_id: int = 0,
    ) -> int:
        payload = (
            f"{self.derivation_version}:{self.root_seed}:{component}:{stage}:{rank}:{worker_id}"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:16], 16) % (2**31)

    def seed_python_random(self, component: str, *, stage: str = "default") -> int:
        seed = self.derive_seed(component, stage=stage)
        random.seed(seed)
        return seed


def apply_global_seed(seed: int, *, component: str = "global") -> dict[str, Any]:
    """Seed available RNG libraries and return applied seed metadata."""

    manager = SeedManager(seed)
    base_seed = manager.seed_python_random(component, stage="runtime")
    applied: dict[str, Any] = {"python_random": base_seed}

    try:
        import numpy as np  # type: ignore[import-not-found]

        np.random.seed(base_seed)
        applied["numpy"] = base_seed
    except Exception:  # pragma: no cover - optional dependency
        applied["numpy"] = "unavailable"

    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        applied["torch"] = base_seed
    except Exception:  # pragma: no cover - optional dependency
        applied["torch"] = "unavailable"

    return applied
