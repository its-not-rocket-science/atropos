"""AWS pricing catalog helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def get_aws_catalog() -> dict[str, Any]:
    """Return an offline AWS pricing catalog in USD."""
    return {
        "ec2": {
            "p4d.24xlarge": {"ondemand_usd_per_hour": 32.77, "spot_usd_per_hour": 12.75},
            "p5.48xlarge": {"ondemand_usd_per_hour": 98.32, "spot_usd_per_hour": 38.10},
            "g4dn.12xlarge": {"ondemand_usd_per_hour": 3.912, "spot_usd_per_hour": 1.58},
            "g5.12xlarge": {"ondemand_usd_per_hour": 5.672, "spot_usd_per_hour": 2.25},
        },
        "sagemaker": {
            "ml.p4d.24xlarge": {"ondemand_usd_per_hour": 36.30},
            "ml.p5.48xlarge": {"ondemand_usd_per_hour": 105.50},
            "ml.g5.12xlarge": {"ondemand_usd_per_hour": 6.85},
        },
        "serverless": {
            "lambda": {
                "usd_per_inference": 0.0000002,
                "usd_per_gb_second": 0.0000166667,
                "notes": (
                    "AWS Lambda GPU is limited/preview; "
                    "modeled as CPU baseline for comparison."
                ),
            }
        },
    }


def fetch_aws_live_catalog(mock: bool = False) -> dict[str, Any]:
    """Return live (or mocked) AWS catalog payload.

    We deliberately keep the default path API-key free and deterministic.
    If ``mock`` is true, this simulates a CI-safe live fetch and annotates metadata.
    """
    catalog = get_aws_catalog()
    source = "aws-live-mock" if mock else "aws-static-fallback"
    catalog["_meta"] = {
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return catalog
