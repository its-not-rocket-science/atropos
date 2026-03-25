# Contributing to Atropos

Thank you for your interest in contributing to Atropos! This document outlines the process for contributing code, documentation, tests, and other improvements to the project.

## Development Setup

### Prerequisites

- Python 3.10 or higher
- Git with LFS support (for handling large test files)
- Basic familiarity with Python development tools (pip, virtual environments)

### Installation

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/your-username/atropos.git
   cd atropos
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix/macOS:
   source .venv/bin/activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e .[dev]
   ```

4. **Set up external pruning frameworks** (optional, for pruning experiments):
   ```bash
   python scripts/setup_wanda.py
   # Or use the Makefile target:
   make setup-wanda
   ```

### Development Commands

Use the Makefile for common development tasks:

| Command | Purpose |
|---------|---------|
| `make install` | Install development dependencies |
| `make test` | Run all tests with coverage reporting |
| `make test-ci` | Run tests excluding integration tests |
| `make lint` | Check code quality with ruff (format and lint) |
| `make typecheck` | Run static type checking with mypy |
| `make format` | Fix linting issues and format code |
| `make clean` | Clean build artifacts and caches |
| `make build` | Build package distributions |
| `make version` | Show current version |

## Coding Standards

### Code Style

Atropos uses **ruff** for linting and formatting with the following configuration:

- **Line length**: 100 characters
- **Python version**: 3.10+
- **Import sorting**: Enforced
- **Unused imports/variables**: Flagged as errors

Run `make lint` to check your code meets these standards before submitting.

### Type Annotations

All code must include type annotations using Python's type hint system. We use **mypy** with strict mode enabled. Run `make typecheck` to verify type correctness.

### Documentation

- **Docstrings**: Use Google-style docstrings for public functions and classes
- **Comments**: Add comments for complex logic, but prefer self-documenting code
- **README/CLAUDE.md**: Keep these files up-to-date with any changes to development workflow

### Project Structure

- `src/atropos/` - Main package source code
- `tests/` - Test files (mirroring source structure)
- `docs/` - User and developer documentation
- `scripts/` - Utility scripts for development and release
- `examples/` - Example scenario files and usage examples
- `external/` - Git submodules for external pruning frameworks

### Key Rules (Non-negotiable)

1. **Never commit code without running tests and linting**
2. **Keep README.md and ROADMAP.md up-to-date**
3. **Ensure ruff check passes** (no missing imports, unused variables/arguments, etc.)
4. **Ensure mypy src passes** (no type errors)
5. **Do not modify the type ignore comment on line 133 of `src/atropos/validation/runner.py`**
6. **Run `ruff format --check` before marking code as done**
7. **Never add "Co-Authored-By:" or similar attribution lines to commit messages**

## Testing

### Test Structure

- Unit tests go in `tests/` directory
- Test files should mirror source structure (e.g., `test_calculations.py` for `calculations.py`)
- Use descriptive test names following `test_<functionality>_<scenario>` pattern
- Mark integration tests with `@pytest.mark.integration` (these are skipped in CI)

### Running Tests

```bash
# Run all tests
make test

# Run a specific test file
pytest tests/test_calculations.py -v

# Run a single test
pytest tests/test_calculations.py::test_estimate_outcome_reduces_energy_for_positive_optimization -v

# Run tests with coverage report
pytest tests/ -v --cov=atropos --cov-report=term-missing
```

### Test Requirements

- **Coverage**: Aim for high test coverage, especially for critical path code
- **Independence**: Tests should be independent and not rely on shared state
- **Speed**: Unit tests should be fast; slow tests should be marked as integration
- **Clarity**: Tests should be readable and clearly demonstrate the expected behavior

### Integration Tests

Integration tests require external dependencies (pruning frameworks, large models) and are skipped in CI. Run them locally with:

```bash
pytest -m integration
```

## Pull Request Workflow

### Creating a Pull Request

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following coding standards

3. **Run tests and linting**:
   ```bash
   make lint
   make typecheck
   make test
   ```

4. **Commit your changes** with a descriptive commit message:
   ```bash
   git add .
   git commit -m "feat: add hyperparameter tuning for pruning targets"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference to any related issues
   - Test plan showing how the changes were verified

### PR Review Process

1. **Automated Checks**: GitHub Actions will run CI tests:
   - Linting (ruff) on Python 3.10, 3.11, 3.12
   - Type checking (mypy) on Python 3.10, 3.11, 3.12
   - Unit tests (pytest) on Python 3.10, 3.11, 3.12
   - Package build verification

2. **Code Review**: Maintainers will review your PR for:
   - Code quality and adherence to standards
   - Test coverage and correctness
   - Documentation updates
   - Performance implications
   - Backward compatibility

3. **Changes Requested**: If changes are needed, update your branch and push again

4. **Approval**: Once approved, a maintainer will merge your PR

### Commit Message Convention

Use conventional commit messages:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or modifications
- `refactor:` Code restructuring
- `chore:` Maintenance tasks
- `ci:` CI/CD changes

## Release Process

### Version Management

- Version numbers follow [Semantic Versioning](https://semver.org/)
- Update version in `pyproject.toml` and `src/atropos/__init__.py`
- Use `make version` to verify consistency

### Release Steps

1. **Update CHANGELOG.md** with changes since last release
2. **Update version numbers** using `scripts/prepare_release.py`
3. **Create release commit and tag**:
   ```bash
   git commit -m "Prepare release vX.Y.Z"
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   ```
4. **Build and verify**:
   ```bash
   make verify-release
   ```
5. **Push tag** to trigger GitHub Actions release workflow
6. **Monitor release workflow** which will:
   - Build distribution packages
   - Upload to PyPI
   - Create GitHub release with changelog

### Pre-release Testing

Before tagging a release, test the package:

```bash
make build
pip install dist/*.whl
python scripts/smoke_test.py
```

## Additional Resources

### Documentation

- [README.md](README.md) - Project overview and quick start
- [CLAUDE.md](CLAUDE.md) - Development instructions for Claude Code
- [ROADMAP.md](ROADMAP.md) - Project roadmap and future plans
- [docs/](docs/) - Detailed user and developer documentation

### External Dependencies

- **Wanda pruning framework**: Located in `external/wanda/` submodule
- **LLM-Pruner/SparseGPT**: Dependencies included in main `pyproject.toml`
- **Pruning framework setup**: See `scripts/setup_wanda.py` for Wanda dependencies

### Getting Help

- **Issues**: Use [GitHub Issues](https://github.com/its-not-rocket-science/atropos/issues) for bug reports and feature requests
- **Discussion**: Use GitHub Discussions for questions and design discussions
- **Contributing questions**: Comment on the PR or tag maintainers

---

Thank you for contributing to Atropos! Your efforts help make LLM optimization more accessible and practical for everyone.