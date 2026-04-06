#!/usr/bin/env python3
"""Refresh offline cloud pricing snapshot used by Atropos."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from atropos.costs.cloud_pricing import CloudPricingEngine


def main() -> int:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    engine = CloudPricingEngine(data_dir=data_dir)
    live = engine.fetch_live_pricing()

    target = data_dir / f"cloud_pricing_{datetime.now(tz=timezone.utc).date().isoformat()}.json"

    if not any(live.values()):
        print("No live pricing data fetched; keeping existing cached dataset.")
        return 0

    # This script intentionally updates only source notes to keep a stable schema.
    # For full updates, maintainers can merge live values into provider entries.
    latest = engine.load_catalog(max_age_days=3650)
    latest["as_of_date"] = datetime.now(tz=timezone.utc).date().isoformat()
    latest["source_notes"] = [
        "Weekly refresh script executed",
        "Optional live pricing fetch performed where public endpoints were available",
    ]
    latest["live_samples"] = live

    target.write_text(json.dumps(latest, indent=2))
    print(f"Wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
