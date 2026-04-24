#!/usr/bin/env python3
"""Platform-core acceptance and invariant checks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKLIST = ROOT / "docs" / "platform_acceptance_checklist.md"
PR_TEMPLATE = ROOT / ".github" / "pull_request_template.md"

CHECKLIST_CATEGORIES = [
    "durable state",
    "idempotent ingestion",
    "health/readiness",
    "structured logs",
    "service metrics",
    "integration tests",
    "restart recovery",
    "deployment artifacts",
    "config validation",
]

REQUIRED_TESTS = [
    ROOT / "tests" / "test_runtime_server_storage.py",
    ROOT / "tests" / "test_api_runtime_e2e_integration.py",
    ROOT / "tests" / "test_structured_logging.py",
    ROOT / "tests" / "test_runtime_controller.py",
    ROOT / "tests" / "test_worker_runtime_observability.py",
]

REQUIRED_DEPLOYMENT_ARTIFACTS = [
    ROOT / "deploy" / "k8s" / "api.yaml",
    ROOT / "deploy" / "k8s" / "worker.yaml",
    ROOT / "deploy" / "k8s" / "configmap.yaml",
    ROOT / "docker" / "Dockerfile.api",
    ROOT / "docker" / "Dockerfile.worker",
    ROOT / "examples" / "runtime-profiles" / "production.env.example",
]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _validate_categories(path: Path, content: str, errors: list[str]) -> None:
    lowered = content.casefold()
    for category in CHECKLIST_CATEGORIES:
        if category not in lowered:
            errors.append(f"{path.relative_to(ROOT)} missing required category: '{category}'.")


def check() -> list[str]:
    errors: list[str] = []

    for required_doc in (CHECKLIST, PR_TEMPLATE):
        if not required_doc.exists():
            errors.append(f"Missing required document: {required_doc.relative_to(ROOT)}.")
            continue

        _validate_categories(required_doc, _read(required_doc), errors)

    for test_file in REQUIRED_TESTS:
        if not test_file.exists():
            errors.append(f"Missing required platform-core test: {test_file.relative_to(ROOT)}.")

    for artifact in REQUIRED_DEPLOYMENT_ARTIFACTS:
        if not artifact.exists():
            errors.append(f"Missing deployment/config artifact: {artifact.relative_to(ROOT)}.")

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("Platform-core invariant check failed:\n", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Platform-core invariant check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
