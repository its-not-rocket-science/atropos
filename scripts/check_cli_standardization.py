#!/usr/bin/env python3
"""Check for atropos command usage in documentation."""

import re
from pathlib import Path


def main():
    repo_root = Path(__file__).parent
    patterns = ["*.md", "*.rst", "*.txt", "*.py"]
    exclude_dirs = [".git", "external", ".venv", "__pycache__", "build", "dist"]

    for pattern in patterns:
        for filepath in repo_root.rglob(pattern):
            # Skip excluded directories
            if any(exclude in str(filepath) for exclude in exclude_dirs):
                continue
            if filepath.is_file():
                try:
                    content = filepath.read_text(encoding="utf-8", errors="ignore")
                    # Look for 'atropos ' (with space) that's likely a command
                    # Exclude 'atropos-llm' and 'atropos.' (like in atropos.cli)
                    matches = re.findall(r"\batropos\b(?![-\w])", content)
                    if matches:
                        print(f"{filepath.relative_to(repo_root)}: {len(matches)} matches")
                        # Show context
                        lines = content.split("\n")
                        for i, line in enumerate(lines, 1):
                            if re.search(r"\batropos\b(?![-\w])", line):
                                print(f"  Line {i}: {line.strip()[:80]}")
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")


if __name__ == "__main__":
    main()
