"""Tests for platform-core invariant checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_INVARIANTS_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_platform_core_invariants.py"
)
_SPEC = importlib.util.spec_from_file_location("check_platform_core_invariants", _INVARIANTS_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load scripts/check_platform_core_invariants.py")

invariants = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(invariants)
ORIGINAL_ROOT = invariants.ROOT
ORIGINAL_REQUIRED_TESTS = invariants.REQUIRED_TESTS
ORIGINAL_REQUIRED_ARTIFACTS = invariants.REQUIRED_DEPLOYMENT_ARTIFACTS


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_required_tree(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "platform_acceptance_checklist.md",
        "\n".join(f"## {category}" for category in invariants.CHECKLIST_CATEGORIES),
    )
    _write(
        tmp_path / ".github" / "pull_request_template.md",
        "\n".join(f"### {category}" for category in invariants.CHECKLIST_CATEGORIES),
    )

    for test_file in ORIGINAL_REQUIRED_TESTS:
        relative = test_file.relative_to(ORIGINAL_ROOT)
        _write(tmp_path / relative, "def test_placeholder() -> None:\n    assert True\n")

    for artifact in ORIGINAL_REQUIRED_ARTIFACTS:
        relative = artifact.relative_to(ORIGINAL_ROOT)
        _write(tmp_path / relative, "placeholder\n")


def _patch_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(invariants, "ROOT", tmp_path)
    monkeypatch.setattr(
        invariants, "CHECKLIST", tmp_path / "docs" / "platform_acceptance_checklist.md"
    )
    monkeypatch.setattr(
        invariants, "PR_TEMPLATE", tmp_path / ".github" / "pull_request_template.md"
    )
    monkeypatch.setattr(
        invariants,
        "REQUIRED_TESTS",
        [tmp_path / path.relative_to(ORIGINAL_ROOT) for path in ORIGINAL_REQUIRED_TESTS],
    )
    monkeypatch.setattr(
        invariants,
        "REQUIRED_DEPLOYMENT_ARTIFACTS",
        [tmp_path / path.relative_to(ORIGINAL_ROOT) for path in ORIGINAL_REQUIRED_ARTIFACTS],
    )


def test_platform_core_invariants_pass(tmp_path: Path, monkeypatch) -> None:
    _seed_required_tree(tmp_path)
    _patch_paths(tmp_path, monkeypatch)

    assert invariants.check() == []


def test_platform_core_invariants_detect_missing_category(tmp_path: Path, monkeypatch) -> None:
    _seed_required_tree(tmp_path)
    checklist_path = tmp_path / "docs" / "platform_acceptance_checklist.md"
    checklist_path.write_text("## durable state\n", encoding="utf-8")

    _patch_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(invariants, "CHECKLIST", checklist_path)

    errors = invariants.check()

    assert any("missing required category" in error for error in errors)
