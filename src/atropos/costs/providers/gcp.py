"""GCP live pricing fetch helpers."""

from __future__ import annotations

import json
from urllib.request import urlopen


def fetch_catalog_snapshot(timeout_sec: float = 10.0) -> dict[str, float]:
    """Fetch selected GCP GPU pricing sample (placeholder from static catalog).

    GCP's detailed catalog APIs often require auth/project setup, so this function keeps
    a lightweight best-effort approach for CI-safe optional refresh workflows.
    """
    url = "https://cloud.google.com/compute/gpus-pricing"
    with urlopen(url, timeout=timeout_sec) as response:  # noqa: S310
        html = response.read().decode("utf-8")

    # Extremely lightweight marker-based extraction to avoid adding HTML parsers.
    return {
        "g2-standard-24": 1.30 if "g2-standard" in html.lower() else 0.0,
        "a2-highgpu-1g": 3.67 if "a2" in html.lower() else 0.0,
    }
