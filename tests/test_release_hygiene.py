"""Tests for repo-local release hygiene checks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_HYGIENE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_release_hygiene.py"
_SPEC = importlib.util.spec_from_file_location("check_release_hygiene", _HYGIENE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load scripts/check_release_hygiene.py")

hygiene = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hygiene)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_release_hygiene_passes_on_consistent_layout(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "atropos-llm"\nversion = "1.2.3"\n')
    _write(tmp_path / "src" / "atropos" / "__init__.py", '__version__ = "1.2.3"\n')
    _write(tmp_path / "CHANGELOG.md", "## [1.2.3] - 2026-01-01\n")
    _write(tmp_path / "README.md", "pip install atropos-llm==1.2.3\natropos-llm --help\n")
    _write(
        tmp_path / "docs" / "installation.md",
        "pip install atropos-llm==1.2.3\nfrom atropos import api\n",
    )
    _write(
        tmp_path / "docs" / "pypi-release-guide.md",
        "Release v<version>\npip install atropos-llm==1.2.3\n",
    )

    monkeypatch.setattr(hygiene, "ROOT", tmp_path)
    monkeypatch.setattr(hygiene, "PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(hygiene, "PACKAGE_INIT", tmp_path / "src" / "atropos" / "__init__.py")
    monkeypatch.setattr(hygiene, "CHANGELOG", tmp_path / "CHANGELOG.md")
    monkeypatch.setattr(hygiene, "RELEASE_GUIDE", tmp_path / "docs" / "pypi-release-guide.md")
    monkeypatch.setattr(
        hygiene,
        "VERSION_SENSITIVE_DOCS",
        [
            tmp_path / "README.md",
            tmp_path / "docs" / "installation.md",
            tmp_path / "docs" / "pypi-release-guide.md",
        ],
    )
    monkeypatch.setattr(
        hygiene,
        "DOC_FILES",
        [tmp_path / "README.md", tmp_path / "docs" / "installation.md"],
    )

    assert hygiene.check() == []


def test_release_hygiene_flags_naming_rule_violation(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "pyproject.toml", '[project]\nname = "atropos-llm"\nversion = "1.2.3"\n')
    _write(tmp_path / "src" / "atropos" / "__init__.py", '__version__ = "1.2.3"\n')
    _write(tmp_path / "CHANGELOG.md", "## [1.2.3] - 2026-01-01\n")
    _write(tmp_path / "README.md", "atropos --help\n")
    _write(tmp_path / "docs" / "installation.md", "")
    _write(tmp_path / "docs" / "pypi-release-guide.md", "Release v<version>\n")

    monkeypatch.setattr(hygiene, "ROOT", tmp_path)
    monkeypatch.setattr(hygiene, "PYPROJECT", tmp_path / "pyproject.toml")
    monkeypatch.setattr(hygiene, "PACKAGE_INIT", tmp_path / "src" / "atropos" / "__init__.py")
    monkeypatch.setattr(hygiene, "CHANGELOG", tmp_path / "CHANGELOG.md")
    monkeypatch.setattr(hygiene, "RELEASE_GUIDE", tmp_path / "docs" / "pypi-release-guide.md")
    monkeypatch.setattr(
        hygiene,
        "VERSION_SENSITIVE_DOCS",
        [
            tmp_path / "README.md",
            tmp_path / "docs" / "installation.md",
            tmp_path / "docs" / "pypi-release-guide.md",
        ],
    )
    monkeypatch.setattr(hygiene, "DOC_FILES", [tmp_path / "README.md"])

    errors = hygiene.check()

    assert any("atropos-llm" in error and "CLI examples" in error for error in errors)
