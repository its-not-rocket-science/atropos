"""Documentation lint checks for import and CLI consistency."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "docs").glob("*.md"))]

INVALID_IMPORT_RE = re.compile(r"^\s*from\s+atropos-llm\s+import\b")
ABTEST_CREATE_RE = re.compile(r"^\s*atropos-llm\s+ab-test\s+create\b")
REQUIRED_CREATE_FLAG = "--config"


def _iter_lines(path: Path) -> list[tuple[int, str]]:
    return [(idx, line) for idx, line in enumerate(path.read_text().splitlines(), start=1)]


def test_docs_do_not_use_invalid_python_package_import() -> None:
    violations: list[str] = []
    for path in DOC_FILES:
        for line_no, line in _iter_lines(path):
            if INVALID_IMPORT_RE.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")

    assert not violations, "Invalid Python import style found:\n" + "\n".join(violations)


def test_ab_test_create_examples_use_config_flag() -> None:
    violations: list[str] = []
    for path in DOC_FILES:
        for line_no, line in _iter_lines(path):
            if ABTEST_CREATE_RE.search(line) and REQUIRED_CREATE_FLAG not in line:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")

    assert not violations, (
        "ab-test create examples must include --config:\n"
        + "\n".join(violations)
    )
