#!/usr/bin/env python3
"""Refresh weekly cloud pricing cache file."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from atropos.costs.cloud_pricing import CloudPricingEngine


def main() -> int:
    parser = argparse.ArgumentParser(description="Update cached cloud pricing JSON")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--fetch-live-pricing", action="store_true")
    args = parser.parse_args()

    engine = CloudPricingEngine(data_dir=args.data_dir)
    if args.fetch_live_pricing:
        engine.refresh_live_pricing()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    out = args.data_dir / f"cloud_pricing_{date.today().isoformat()}.json"
    out.write_text(json.dumps(engine.catalog, indent=2, sort_keys=True))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
