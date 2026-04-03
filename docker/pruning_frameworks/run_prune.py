"""Minimal container entrypoint for framework-specific prune execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    out_dir = Path(args.save)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).write_text(
        json.dumps({"framework": args.framework, "status": "completed", "mode": "container"}),
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text(
        json.dumps({"framework": args.framework, "model": args.model, "sparsity": args.sparsity}),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
