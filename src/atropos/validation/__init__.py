"""Validation framework for testing Atropos against real neural networks."""

from __future__ import annotations

__all__ = [
    "ModelValidator",
    "ValidationResult",
    "run_validation",
]

from .models import ValidationResult
from .runner import ModelValidator, run_validation
