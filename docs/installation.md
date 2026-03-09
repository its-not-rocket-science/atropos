# Installation

## Requirements

- Python 3.10 or higher

## Install from Source

```bash
git clone https://github.com/its-not-rocket-science/atropos.git
cd atropos
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

## Install Development Dependencies

```bash
pip install -e .[dev]
```

This installs:
- pytest for testing
- ruff for linting
- mypy for type checking

## Verify Installation

```bash
atropos --help
```

## Update

```bash
git pull
pip install -e .
```
