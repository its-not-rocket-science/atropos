#!/usr/bin/env python3
"""Repo-local release hygiene checks.

This script intentionally stays deterministic and does not query external services.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
PACKAGE_INIT = ROOT / "src" / "atropos" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"
RELEASE_GUIDE = ROOT / "docs" / "pypi-release-guide.md"
VERSION_SENSITIVE_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "installation.md",
    ROOT / "docs" / "pypi-release-guide.md",
]
DOC_FILES = [
    ROOT / "README.md",
    *(
        path
        for path in (ROOT / "docs").glob("*.md")
        if "audit" not in path.name and "assessment" not in path.name
    ),
]

SEMVER_RE = re.compile(r"\b(\d+\.\d+\.\d+)\b")
PLACEHOLDER_VERSION = "<version>"
INVALID_IMPORT_PATTERNS = (
    re.compile(r"\bfrom\s+atropos-llm\s+import\b"),
    re.compile(r"\bimport\s+atropos-llm\b"),
)
CLI_LINE_RE = re.compile(r"^\s*atropos(?:\s|$)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_line_numbers(path: Path, pattern: re.Pattern[str]) -> list[int]:
    matches: list[int] = []
    for idx, line in enumerate(_read(path).splitlines(), start=1):
        if pattern.search(line):
            matches.append(idx)
    return matches


def _pyproject_version() -> str:
    content = _read(PYPROJECT)
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not locate [project].version in pyproject.toml")
    return match.group(1)


def _package_version() -> str:
    content = _read(PACKAGE_INIT)
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', content, flags=re.MULTILINE)
    if not match:
        raise RuntimeError("Could not locate __version__ in src/atropos/__init__.py")
    return match.group(1)


def check() -> list[str]:
    errors: list[str] = []

    pyproject_version = _pyproject_version()
    package_version = _package_version()
    if pyproject_version != package_version:
        errors.append(
            "Version mismatch: pyproject.toml has "
            f"{pyproject_version}, src/atropos/__init__.py has {package_version}."
        )

    changelog_content = _read(CHANGELOG)
    changelog_header = re.compile(rf"^##\s+\[{re.escape(pyproject_version)}\]", re.MULTILINE)
    if not changelog_header.search(changelog_content):
        errors.append(
            f"CHANGELOG.md does not contain a top-level release section for [{pyproject_version}]."
        )

    # The most recent concrete release section should match pyproject version.
    released_versions = re.findall(r"^##\s+\[(\d+\.\d+\.\d+)\]", changelog_content, re.MULTILINE)
    if released_versions and released_versions[0] != pyproject_version:
        errors.append(
            "Latest CHANGELOG.md release does not match pyproject version: "
            f"{released_versions[0]} != {pyproject_version}."
        )

    release_guide_versions = set(SEMVER_RE.findall(_read(RELEASE_GUIDE)))
    release_guide_versions.discard(pyproject_version)
    if release_guide_versions:
        errors.append(
            "docs/pypi-release-guide.md contains hard-coded version literals "
            f"{sorted(release_guide_versions)}; use {PLACEHOLDER_VERSION} for examples."
        )

    for path in VERSION_SENSITIVE_DOCS:
        content = _read(path)
        if "atropos-llm==" in content and f"atropos-llm=={pyproject_version}" not in content:
            errors.append(
                f"{path.relative_to(ROOT)} has a pinned atropos-llm install example that is "
                f"not {pyproject_version}."
            )

    for path in DOC_FILES:
        lines = _read(path).splitlines()
        for line_no, line in enumerate(lines, start=1):
            if any(pattern.search(line) for pattern in INVALID_IMPORT_PATTERNS):
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no} invalid Python import; "
                    "use `import atropos`."
                )

            if CLI_LINE_RE.match(line):
                errors.append(
                    f"{path.relative_to(ROOT)}:{line_no} CLI examples must use "
                    "`atropos-llm`, not `atropos`."
                )

    return errors


def main() -> int:
    errors = check()
    if errors:
        print("Release hygiene check failed:\n", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Release hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
