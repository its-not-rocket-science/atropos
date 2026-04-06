"""Azure pricing catalog helpers."""

from __future__ import annotations

from typing import Any


def get_azure_catalog() -> dict[str, Any]:
    """Return a static Azure pricing catalog in USD."""
    return {
        "vm": {
            "Standard_NC4as_T4_v3": {"ondemand_usd_per_hour": 0.90, "spot_usd_per_hour": 0.37},
            "Standard_NC24ads_A100_v4": {
                "ondemand_usd_per_hour": 14.69,
                "spot_usd_per_hour": 5.65,
            },
            "Standard_NC96ads_A100_v4": {
                "ondemand_usd_per_hour": 58.76,
                "spot_usd_per_hour": 22.25,
            },
        },
        "ml_endpoints": {
            "ManagedOnlineEndpoint_NC24ads_A100_v4": {"ondemand_usd_per_hour": 16.2}
        },
    }
