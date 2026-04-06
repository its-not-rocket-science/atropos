"""Azure pricing catalog helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def get_azure_catalog() -> dict[str, Any]:
    """Return an offline Azure pricing catalog in USD."""
    return {
        "vm": {
            "Standard_NC4as_T4_v3": {"ondemand_usd_per_hour": 0.90, "spot_usd_per_hour": 0.37},
            "Standard_NC24_A100_v2": {"ondemand_usd_per_hour": 12.40, "spot_usd_per_hour": 4.95},
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
            "ManagedOnlineEndpoint_NC24_A100_v2": {"ondemand_usd_per_hour": 13.70},
            "ManagedOnlineEndpoint_NC24ads_A100_v4": {"ondemand_usd_per_hour": 16.2},
        },
    }


def fetch_azure_live_catalog(mock: bool = False) -> dict[str, Any]:
    """Return live (or mocked) Azure catalog payload."""
    catalog = get_azure_catalog()
    source = "azure-live-mock" if mock else "azure-static-fallback"
    catalog["_meta"] = {
        "source": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return catalog
