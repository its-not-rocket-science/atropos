#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "[atropos] migrating pruning integration assets..."

mkdir -p artifacts/pruning/wanda artifacts/pruning/sparsegpt artifacts/pruning/llm-pruner

if command -v docker >/dev/null 2>&1; then
  RUNTIME="docker"
elif command -v podman >/dev/null 2>&1; then
  RUNTIME="podman"
else
  echo "No docker/podman runtime detected. Please install rootless docker or podman." >&2
  exit 1
fi

echo "Using runtime: $RUNTIME"

if [[ "${1:-}" == "--rebuild" ]]; then
  "$RUNTIME" compose -f docker-compose.yml build --no-cache
else
  "$RUNTIME" compose -f docker-compose.yml build
fi

echo "Migration complete. Next steps:"
echo "  1) atropos-llm setup-pruning"
echo "  2) atropos-llm test-pruning --framework wanda"
