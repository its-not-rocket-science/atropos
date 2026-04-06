"""AWS live pricing fetch helpers."""

from __future__ import annotations

import json
from urllib.request import urlopen


def fetch_ec2_on_demand_samples(timeout_sec: float = 10.0) -> dict[str, float]:
    """Fetch sample EC2 prices from AWS public pricing API.

    Returns a mapping of instance type to USD/hour when available.
    """
    # Public JSON index for on-demand pricing by region.
    url = (
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/"
        "us-east-1/index.json"
    )
    with urlopen(url, timeout=timeout_sec) as response:  # noqa: S310
        payload = json.loads(response.read().decode("utf-8"))

    products = payload.get("products", {})
    terms = payload.get("terms", {}).get("OnDemand", {})
    prices: dict[str, float] = {}

    for sku, product in products.items():
        attributes = product.get("attributes", {})
        instance_type = attributes.get("instanceType")
        if instance_type not in {"p4d.24xlarge", "p5.48xlarge", "g4dn.xlarge", "g5.2xlarge"}:
            continue

        sku_terms = terms.get(sku, {})
        for term in sku_terms.values():
            for dimension in term.get("priceDimensions", {}).values():
                usd = dimension.get("pricePerUnit", {}).get("USD")
                if usd:
                    prices[instance_type] = float(usd)
                    break
            if instance_type in prices:
                break

    return prices
