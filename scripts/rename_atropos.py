#!/usr/bin/env python3
"""Replace 'atropos ' with 'atropos-llm ' in documentation files."""

from pathlib import Path


def replace_in_file(filepath: Path) -> None:
    """Replace atropos commands in a file."""
    try:
        content = filepath.read_text(encoding="utf-8")
        # Replace 'atropos ' (with space) with 'atropos-llm ' (keep space)
        # This ensures we don't replace 'atropos' in URLs or other contexts
        new_content = content.replace("atropos ", "atropos-llm ")
        if new_content != content:
            filepath.write_text(new_content, encoding="utf-8")
            print(f"Updated {filepath}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")


def main() -> None:
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs"

    for filepath in docs_dir.glob("*.md"):
        replace_in_file(filepath)

    print("Done.")


if __name__ == "__main__":
    main()
