"""Azure live pricing fetch helpers."""

from __future__ import annotations

import json
from urllib.parse import quote
from urllib.request import urlopen


def fetch_vm_prices(region: str = "eastus", timeout_sec: float = 10.0) -> dict[str, float]:
    """Fetch selected Azure VM retail prices in USD/hour."""
    filter_expr = (
        "$filter=serviceName eq 'Virtual Machines' and "
        f"armRegionName eq '{region}' and "
        "(armSkuName eq 'Standard_NC4as_T4_v3' or "
        "armSkuName eq 'Standard_NC24ads_A100_v4')"
    )
    quoted_filter = quote(filter_expr, safe="=&' ()")
    url = f"https://prices.azure.com/api/retail/prices?{quoted_filter}"

    with urlopen(url, timeout=timeout_sec) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    out: dict[str, float] = {}
    for item in payload.get("Items", []):
        sku = item.get("armSkuName")
        if sku == "Standard_NC4as_T4_v3":
            out["ncas_t4_v3"] = float(item.get("retailPrice", 0.0))
        elif sku == "Standard_NC24ads_A100_v4":
            out["nc_a100_v4"] = float(item.get("retailPrice", 0.0))
    return out
