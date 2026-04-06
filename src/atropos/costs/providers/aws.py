"""AWS pricing catalog helpers.

Live APIs are intentionally optional; this module always provides static defaults.
"""

from __future__ import annotations

from typing import Any


def get_aws_catalog() -> dict[str, Any]:
    """Return a static AWS pricing catalog in USD."""
    return {
        "ec2": {
            "p4d.24xlarge": {"ondemand_usd_per_hour": 32.77, "spot_usd_per_hour": 12.75},
            "p5.48xlarge": {"ondemand_usd_per_hour": 98.32, "spot_usd_per_hour": 38.10},
            "g4dn.12xlarge": {"ondemand_usd_per_hour": 3.912, "spot_usd_per_hour": 1.58},
            "g5.12xlarge": {"ondemand_usd_per_hour": 5.672, "spot_usd_per_hour": 2.25},
        },
        "sagemaker": {
            "ml.p4d.24xlarge": {"ondemand_usd_per_hour": 36.30},
            "ml.g5.12xlarge": {"ondemand_usd_per_hour": 6.85},
        },
        "serverless": {
            "lambda": {
                "usd_per_million_requests": 0.20,
                "usd_per_gb_second": 0.0000166667,
                "notes": "GPU Lambda is not broadly available; modeled as CPU/serverless baseline.",
            }
        },
    }
