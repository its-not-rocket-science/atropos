"""GCP pricing catalog helpers."""

from __future__ import annotations

from typing import Any


def get_gcp_catalog() -> dict[str, Any]:
    """Return a static GCP pricing catalog in USD."""
    return {
        "compute_engine": {
            "g2-standard-24-l4": {"ondemand_usd_per_hour": 2.34, "spot_usd_per_hour": 0.79},
            "a2-highgpu-1g": {"ondemand_usd_per_hour": 3.67, "spot_usd_per_hour": 1.17},
            "a2-highgpu-8g": {"ondemand_usd_per_hour": 29.36, "spot_usd_per_hour": 9.36},
        },
        "vertex_ai": {
            "n1-standard-16-plus-t4": {"ondemand_usd_per_hour": 1.95},
            "a2-highgpu-1g": {"ondemand_usd_per_hour": 4.35},
        },
    }
