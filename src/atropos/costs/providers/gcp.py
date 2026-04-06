"""GCP pricing catalog helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def get_gcp_catalog() -> dict[str, Any]:
    """Return an offline GCP pricing catalog in USD."""
    return {
        "compute_engine": {
            "g2-standard-24-l4": {"ondemand_usd_per_hour": 2.34, "spot_usd_per_hour": 0.79},
            "a2-highgpu-1g": {"ondemand_usd_per_hour": 3.67, "spot_usd_per_hour": 1.17},
            "a2-highgpu-8g": {"ondemand_usd_per_hour": 29.36, "spot_usd_per_hour": 9.36},
        },
        "vertex_ai": {
            "vertex-a2-highgpu-1g": {"ondemand_usd_per_hour": 4.35},
            "vertex-g2-standard-24-l4": {"ondemand_usd_per_hour": 2.70},
        },
    }


def fetch_gcp_live_catalog(mock: bool = False) -> dict[str, Any]:
    """Return live (or mocked) GCP catalog payload."""
    catalog = get_gcp_catalog()
    source = "gcp-live-mock" if mock else "gcp-static-fallback"
    catalog["_meta"] = {
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return catalog
