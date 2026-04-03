"""Calibration utility for quality degradation predictions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CalibrationFit:
    """Fitted affine calibration coefficients."""

    scale: float
    offset: float
    num_samples: int


def _extract_degradation(record: dict[str, Any]) -> tuple[float, float] | None:
    actual = record.get("actual", {}) or {}
    predicted = record.get("predicted", {}) or {}

    actual_deg = actual.get("quality_degradation_percent")
    pred_deg = predicted.get("quality_degradation_percent")

    if actual_deg is None or pred_deg is None:
        return None
    return float(pred_deg), float(actual_deg)


def fit_affine_calibration(pairs: list[tuple[float, float]]) -> CalibrationFit:
    """Fit actual ≈ scale * predicted + offset via least squares."""
    if not pairs:
        return CalibrationFit(scale=1.0, offset=0.0, num_samples=0)

    x_mean = sum(x for x, _ in pairs) / len(pairs)
    y_mean = sum(y for _, y in pairs) / len(pairs)
    denom = sum((x - x_mean) ** 2 for x, _ in pairs)
    if denom == 0:
        return CalibrationFit(scale=1.0, offset=y_mean - x_mean, num_samples=len(pairs))

    scale = sum((x - x_mean) * (y - y_mean) for x, y in pairs) / denom
    offset = y_mean - scale * x_mean
    return CalibrationFit(scale=scale, offset=offset, num_samples=len(pairs))


def load_validation_pairs(validation_dir: Path) -> list[tuple[float, float]]:
    """Load predicted/actual quality degradation pairs from JSON results."""
    pairs: list[tuple[float, float]] = []
    for path in validation_dir.glob("*.json"):
        if path.name == "suite_summary.json":
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        pair = _extract_degradation(data)
        if pair:
            pairs.append(pair)
    return pairs


def run_calibration(
    validation_dir: Path = Path("validation_results"),
    output_path: Path = Path("configs/quality_calibration.yaml"),
) -> CalibrationFit:
    """Run calibration and persist coefficients to a YAML config file."""
    pairs = load_validation_pairs(validation_dir)
    fit = fit_affine_calibration(pairs)

    payload = {
        "quality_calibration": {
            "formula": "predicted = raw * scale + offset",
            "scale": round(fit.scale, 6),
            "offset": round(fit.offset, 6),
            "num_samples": fit.num_samples,
            "source_dir": str(validation_dir),
        }
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return fit


if __name__ == "__main__":
    fit = run_calibration()
    print(
        f"Calibration completed: scale={fit.scale:.4f}, offset={fit.offset:.4f}, "
        f"samples={fit.num_samples}"
    )
