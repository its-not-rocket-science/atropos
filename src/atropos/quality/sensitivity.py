"""Sensitivity feature engineering for pruning quality prediction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable


@dataclass(frozen=True)
class LayerSensitivity:
    """Sensitivity statistics for a model layer."""

    name: str
    gradient_magnitude: float
    hessian_trace: float
    attention_head_importance: float | None = None
    embedding_fragility: float | None = None

    @property
    def combined(self) -> float:
        """Combined sensitivity score used by quality predictors."""
        head = self.attention_head_importance or 0.0
        fragility = self.embedding_fragility or 0.0
        return (
            0.45 * self.gradient_magnitude
            + 0.35 * self.hessian_trace
            + 0.15 * head
            + 0.05 * fragility
        )


@dataclass(frozen=True)
class SensitivityProfile:
    """Model-level sensitivity profile."""

    layers: tuple[LayerSensitivity, ...]

    @property
    def average_sensitivity(self) -> float:
        """Average combined sensitivity across layers."""
        if not self.layers:
            return 0.0
        return mean(layer.combined for layer in self.layers)

    @property
    def embedding_output_fragility(self) -> float:
        """Mean fragility across embedding/output layers."""
        fragilities = [l.embedding_fragility for l in self.layers if l.embedding_fragility is not None]
        if not fragilities:
            return 0.0
        return mean(fragilities)


def _flatten(values: Iterable[Any]) -> list[float]:
    flat: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            flat.extend(_flatten(value))
        else:
            flat.append(float(value))
    return flat


def gradient_magnitude(gradients: Iterable[Any]) -> float:
    """Compute average absolute gradient magnitude."""
    flat = _flatten(gradients)
    if not flat:
        return 0.0
    return sum(abs(x) for x in flat) / len(flat)


def hessian_trace_approximation(second_order_diagonal: Iterable[Any]) -> float:
    """Approximate Hessian trace from a diagonal estimate (OBD style)."""
    flat = _flatten(second_order_diagonal)
    if not flat:
        return 0.0
    return sum(abs(x) for x in flat)


def attention_head_importance(head_scores: Iterable[float]) -> float:
    """Aggregate attention-head importance from pruning probes."""
    scores = [max(0.0, float(x)) for x in head_scores]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def embedding_output_fragility(weight_norm_delta: float, loss_delta: float) -> float:
    """Estimate fragility for embedding/output layers.

    Larger drift in weight norms and loss indicates pruning-unfriendly regions.
    """
    norm_term = max(0.0, float(weight_norm_delta))
    loss_term = max(0.0, float(loss_delta))
    return math.sqrt(norm_term**2 + loss_term**2)


def build_profile(layer_stats: list[dict[str, Any]]) -> SensitivityProfile:
    """Build a sensitivity profile from collected per-layer stats."""
    layers = tuple(
        LayerSensitivity(
            name=str(layer.get("name", f"layer_{idx}")),
            gradient_magnitude=float(layer.get("gradient_magnitude", 0.0)),
            hessian_trace=float(layer.get("hessian_trace", 0.0)),
            attention_head_importance=(
                float(layer["attention_head_importance"])
                if layer.get("attention_head_importance") is not None
                else None
            ),
            embedding_fragility=(
                float(layer["embedding_fragility"])
                if layer.get("embedding_fragility") is not None
                else None
            ),
        )
        for idx, layer in enumerate(layer_stats)
    )
    return SensitivityProfile(layers=layers)
