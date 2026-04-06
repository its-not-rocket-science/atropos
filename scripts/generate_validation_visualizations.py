#!/usr/bin/env python3
"""Generate simple SVG validation charts without external plotting deps."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


WIDTH = 900
HEIGHT = 520
MARGIN = 60


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_float(value: str | None) -> float | None:
    if value in (None, "", "None", "nan"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def write_svg(path: Path, title: str, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{WIDTH/2}" y="30" text-anchor="middle" font-size="20" font-family="Arial">{title}</text>',
                body,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def scale(values: Iterable[float], lo: float, hi: float, pixel_lo: float, pixel_hi: float) -> list[float]:
    vals = list(values)
    if hi == lo:
        return [(pixel_lo + pixel_hi) / 2 for _ in vals]
    return [pixel_lo + (v - lo) * (pixel_hi - pixel_lo) / (hi - lo) for v in vals]


def scatter_pred_vs_actual(rows: list[dict[str, str]], output: Path) -> None:
    pts = [
        (to_float(r.get("pred_memory_gb")), to_float(r.get("actual_memory_gb")))
        for r in rows
    ]
    pts = [(x, y) for x, y in pts if x is not None and y is not None]

    if not pts:
        body = '<text x="450" y="260" text-anchor="middle" font-size="16">No successful runs with actual memory metrics.</text>'
        write_svg(output, "Predicted vs Actual Memory (GB)", body)
        return

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lo, hi = min(xs + ys), max(xs + ys)
    sx = scale(xs, lo, hi, MARGIN, WIDTH - MARGIN)
    sy = scale(ys, lo, hi, HEIGHT - MARGIN, MARGIN)

    body = [
        f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{WIDTH-MARGIN}" y2="{MARGIN}" stroke="#888" stroke-dasharray="6,4"/>',
        f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{WIDTH-MARGIN}" y2="{HEIGHT-MARGIN}" stroke="black"/>',
        f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{MARGIN}" y2="{MARGIN}" stroke="black"/>',
    ]
    for x, y in zip(sx, sy, strict=True):
        body.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#1f77b4"/>')
    write_svg(output, "Predicted vs Actual Memory (GB)", "\n".join(body))


def throughput_error_bars(rows: list[dict[str, str]], output: Path) -> None:
    vals = []
    for r in rows:
        p = to_float(r.get("pred_throughput_toks_per_sec"))
        a = to_float(r.get("actual_throughput_toks_per_sec"))
        if p is not None and a is not None:
            vals.append((r.get("run_id", "run"), p, abs(a - p)))

    if not vals:
        body = '<text x="450" y="260" text-anchor="middle" font-size="16">No successful runs with actual throughput metrics.</text>'
        write_svg(output, "Throughput Prediction Error Bars", body)
        return

    max_y = max(v[1] + v[2] for v in vals) * 1.1
    body = [
        f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{WIDTH-MARGIN}" y2="{HEIGHT-MARGIN}" stroke="black"/>',
        f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{MARGIN}" y2="{MARGIN}" stroke="black"/>',
    ]
    spacing = (WIDTH - 2 * MARGIN) / max(len(vals), 1)
    for i, (name, pred, err) in enumerate(vals):
        x = MARGIN + spacing * (i + 0.5)
        y = HEIGHT - MARGIN - (pred / max_y) * (HEIGHT - 2 * MARGIN)
        yerr = (err / max_y) * (HEIGHT - 2 * MARGIN)
        body.extend(
            [
                f'<line x1="{x}" y1="{y-yerr}" x2="{x}" y2="{y+yerr}" stroke="#d62728"/>',
                f'<circle cx="{x}" cy="{y}" r="4" fill="#1f77b4"/>',
                f'<text x="{x}" y="{HEIGHT-MARGIN+18}" text-anchor="middle" font-size="9">{name}</text>',
            ]
        )
    write_svg(output, "Throughput Prediction Error Bars", "\n".join(body))


def calibration(rows: list[dict[str, str]], output: Path) -> None:
    total = len(rows)
    success = sum(1 for r in rows if r.get("status") == "ok")
    observed = success / total if total else 0
    pred_prob = 0.9

    x = MARGIN + pred_prob * (WIDTH - 2 * MARGIN)
    y = HEIGHT - MARGIN - observed * (HEIGHT - 2 * MARGIN)
    body = "\n".join(
        [
            f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{WIDTH-MARGIN}" y2="{MARGIN}" stroke="#888" stroke-dasharray="6,4"/>',
            f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{WIDTH-MARGIN}" y2="{HEIGHT-MARGIN}" stroke="black"/>',
            f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{MARGIN}" y2="{MARGIN}" stroke="black"/>',
            f'<circle cx="{x}" cy="{y}" r="6" fill="#2ca02c"/>',
            f'<text x="{x}" y="{y-12}" text-anchor="middle" font-size="12">({pred_prob:.2f}, {observed:.2f})</text>',
        ]
    )
    write_svg(output, "Calibration: Predicted Confidence vs Observed Success", body)


def break_even(rows: list[dict[str, str]], output: Path) -> None:
    preds = [to_float(r.get("pred_break_even_months")) for r in rows]
    preds = [p for p in preds if p is not None]

    if not preds:
        body = '<text x="450" y="260" text-anchor="middle" font-size="16">No predicted break-even values available.</text>'
        write_svg(output, "Break-even Timeline Comparison", body)
        return

    max_y = max(preds) * 1.1
    spacing = (WIDTH - 2 * MARGIN) / max(len(preds), 1)
    body = [
        f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{WIDTH-MARGIN}" y2="{HEIGHT-MARGIN}" stroke="black"/>',
        f'<line x1="{MARGIN}" y1="{HEIGHT-MARGIN}" x2="{MARGIN}" y2="{MARGIN}" stroke="black"/>',
    ]
    path_points = []
    for i, p in enumerate(preds):
        x = MARGIN + spacing * (i + 0.5)
        y = HEIGHT - MARGIN - (p / max_y) * (HEIGHT - 2 * MARGIN)
        path_points.append(f"{x},{y}")
        body.append(f'<circle cx="{x}" cy="{y}" r="4" fill="#1f77b4"/>')
    body.append(f'<polyline points="{" ".join(path_points)}" fill="none" stroke="#1f77b4"/>')

    actuals = [to_float(r.get("actual_break_even_months")) for r in rows]
    if all(a is None for a in actuals):
        body.append('<text x="450" y="470" text-anchor="middle" font-size="14">Actual break-even unavailable (all runs failed).</text>')

    write_svg(output, "Break-even Timeline Comparison", "\n".join(body))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate case-study visualizations")
    parser.add_argument("--input", default="validation_results/validation_2026_04/validation_runs.csv")
    parser.add_argument("--output-dir", default="docs/case_studies/assets/validation_2026_04")
    args = parser.parse_args()

    rows = load_rows(Path(args.input))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scatter_pred_vs_actual(rows, out / "pred_vs_actual_memory.svg")
    throughput_error_bars(rows, out / "throughput_error_bars.svg")
    calibration(rows, out / "calibration_curve.svg")
    break_even(rows, out / "break_even_timeline.svg")
    print(f"Generated 4 SVG charts in {out}")


if __name__ == "__main__":
    main()
