"""Quality degradation predictor for pruning plans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from random import Random
from statistics import mean
from typing import Literal

from .sensitivity import SensitivityProfile

PredictionMethod = Literal["linear", "obd_obs", "lookup"]
UncertaintyMethod = Literal["bootstrap", "quantile", "conformal"]
QualityMetric = Literal["perplexity", "humaneval_pass_at_1", "custom"]


@dataclass(frozen=True)
class CalibrationCoefficients:
    """Affine calibration mapping for model predictions."""

    scale: float = 1.0
    offset: float = 0.0


@dataclass(frozen=True)
class QualityPrediction:
    """Predicted degradation summary."""

    metric: QualityMetric
    degradation_percent: float
    lower_percent: float
    upper_percent: float
    expected_quality: float
    risk_band: Literal["low", "medium", "high"]


@dataclass(frozen=True)
class QualityPredictorConfig:
    """Configuration for quality degradation prediction."""

    method: PredictionMethod = "linear"
    uncertainty_method: UncertaintyMethod = "bootstrap"
    calibration: CalibrationCoefficients = CalibrationCoefficients()
    confidence_level: float = 0.90
    lookup_table: dict[float, float] | None = None


def _risk_from_degradation(percent: float) -> Literal["low", "medium", "high"]:
    if percent < 10:
        return "low"
    if percent < 30:
        return "medium"
    return "high"


def _interpolate_lookup(lookup_table: dict[float, float], sparsity: float) -> float:
    if not lookup_table:
        return sparsity * 100.0

    points = sorted((float(k), float(v)) for k, v in lookup_table.items())
    if sparsity <= points[0][0]:
        return points[0][1]
    if sparsity >= points[-1][0]:
        return points[-1][1]

    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        if x0 <= sparsity <= x1:
            alpha = (sparsity - x0) / (x1 - x0)
            return y0 + alpha * (y1 - y0)
    return points[-1][1]


def _point_prediction(
    method: PredictionMethod,
    sparsity: float,
    sensitivity: float,
    lookup: dict[float, float] | None,
) -> float:
    if method == "linear":
        return 100.0 * sparsity * max(0.05, sensitivity)
    if method == "obd_obs":
        # OBD/OBS-inspired quadratic approximation from second-order Taylor loss.
        return 50.0 * (sparsity**2) * max(0.05, sensitivity)
    return _interpolate_lookup(lookup or {}, sparsity)


def _uncertainty_interval(
    method: UncertaintyMethod,
    point: float,
    sensitivity: float,
    confidence_level: float,
) -> tuple[float, float]:
    spread = max(1.0, point * (0.20 + 0.35 * sensitivity))
    q = max(0.01, min(0.49, (1.0 - confidence_level) / 2.0))

    if method == "quantile":
        return max(0.0, point - spread * (1 - q)), point + spread * (1 - q)
    if method == "conformal":
        radius = spread * 1.1
        return max(0.0, point - radius), point + radius

    rng = Random(1337)
    samples = [max(0.0, rng.gauss(point, spread / 2.0)) for _ in range(250)]
    samples.sort()
    lo_idx = int(len(samples) * q)
    hi_idx = int(len(samples) * (1 - q)) - 1
    return samples[lo_idx], samples[max(hi_idx, lo_idx)]


def predict_quality_degradation(
    *,
    metric: QualityMetric,
    sparsity: float,
    sensitivity_profile: SensitivityProfile,
    baseline_quality: float,
    predictor_config: QualityPredictorConfig,
    custom_metric_callback: Callable[[float], float] | None = None,
) -> QualityPrediction:
    """Predict quality degradation and uncertainty without full evaluation."""
    sensitivity = sensitivity_profile.average_sensitivity
    raw = _point_prediction(
        predictor_config.method,
        sparsity,
        sensitivity,
        predictor_config.lookup_table,
    )
    calibrated = raw * predictor_config.calibration.scale + predictor_config.calibration.offset
    lower, upper = _uncertainty_interval(
        predictor_config.uncertainty_method,
        calibrated,
        sensitivity,
        predictor_config.confidence_level,
    )

    if metric == "custom" and custom_metric_callback:
        expected_quality = custom_metric_callback(calibrated)
    else:
        expected_quality = max(0.0, baseline_quality * (1 - calibrated / 100.0))

    return QualityPrediction(
        metric=metric,
        degradation_percent=calibrated,
        lower_percent=lower,
        upper_percent=upper,
        expected_quality=expected_quality,
        risk_band=_risk_from_degradation(calibrated),
    )


def expected_quality_from_risk(quality_risk: str) -> float:
    """Fallback mapping for existing strategy risk labels."""
    return {"low": 0.90, "medium": 0.75, "high": 0.60}.get(quality_risk, 0.60)


def blend_predictions(values: list[float]) -> float:
    """Average predictions from multiple methods for ensembling."""
    if not values:
        return 0.0
    return mean(values)
