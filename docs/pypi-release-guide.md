# PyPI Release Guide for Maintainers
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.


This guide documents the process for releasing Atropos to PyPI and Test PyPI.

## Overview

Atropos uses GitHub Actions to automatically build and publish packages when tags are pushed. The release pipeline includes:

1. **Testing** – Run full test suite, linting, and type checking
2. **Building** – Create source distribution and wheel
3. **Verification** – Install built package and run smoke tests
4. **Publishing** – Upload to Test PyPI (for testing) and PyPI (for production)

## GitHub Secrets Setup

To enable publishing, you need to configure the following secrets in your GitHub repository:

### For PyPI (production)
1. Go to [PyPI Account Settings > API tokens](https://pypi.org/manage/account/token/)
2. Create a new API token with scope "Entire account" (for new projects) or restrict to the "atropos-llm" project
3. Copy the token value (it will only be shown once)
4. In your GitHub repository: Settings → Secrets and variables → Actions → New repository secret
5. Name: `PYPI_API_TOKEN`
6. Value: Paste the token

### For Test PyPI (testing)
1. Go to [TestPyPI Account Settings > API tokens](https://test.pypi.org/manage/account/token/)
2. Create a new API token with appropriate scope
3. Copy the token value
4. Create a new GitHub secret named `TEST_PYPI_API_TOKEN`

## Environments in GitHub Actions

The release workflow uses GitHub environments for additional security and approval gates:

1. **test-pypi** – For publishing to Test PyPI
2. **pypi** – For publishing to production PyPI

To set up environments:

1. Go to your repository Settings → Environments
2. Click "New environment"
3. Name: `test-pypi`
4. Add the required secret: `TEST_PYPI_API_TOKEN`
5. Repeat for `pypi` environment with `PYPI_API_TOKEN`

## Release Process

### 1. Prepare the Release

```bash
# Update version in pyproject.toml and src/atropos/__init__.py
# Run the release preparation script
python scripts/prepare_release.py

# Update CHANGELOG.md with changes for the new version
# Commit changes
git commit -am "Prepare release v<version>"
```

### 2. Create and Push Tag

```bash
# Create an annotated tag
git tag -a v<version> -m "Release v<version>"

# Push the tag (triggers GitHub Actions)
git push origin v<version>
```

### 3. Monitor the Release Workflow

1. Go to the GitHub repository → Actions
2. Watch the "Release Build and Publish" workflow
3. Verify all jobs pass:
   - `test` – Tests, linting, type checking
   - `build` – Package building
   - `verify` – Smoke test installation
   - `publish-test-pypi` – Upload to Test PyPI (if stable tag)
   - `publish-pypi` – Upload to PyPI (if stable tag)

### 4. Verify the Release

**Test PyPI:**
```bash
pip install --index-url https://test.pypi.org/simple/ atropos-llm
```

**Production PyPI:**
```bash
pip install atropos-llm
```

Visit:
- https://test.pypi.org/project/atropos-llm/ (Test PyPI)
- https://pypi.org/project/atropos-llm/ (Production PyPI)

### 5. Create GitHub Release

1. Go to repository → Releases
2. Click "Draft a new release"
3. Select the tag you pushed
4. Add release notes (copy from CHANGELOG.md)
5. Publish the release

## Manual Publishing (Fallback)

If GitHub Actions fails, you can publish manually:

```bash
# Install build tools
pip install build twine

# Build packages
python -m build

# Upload to Test PyPI
twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## Version Management

- Follow [Semantic Versioning](https://semver.org/)
- Update version in two places:
  1. `pyproject.toml` – `version = "x.y.z"`
  2. `src/atropos/__init__.py` – `__version__ = "x.y.z"`
- Use the `prepare_release.py` script to validate version consistency

## Troubleshooting

### "Repository requires trusted publishing"

The workflow uses PyPI's trusted publishing (OIDC). Ensure:
- The `pypa/gh-action-pypi-publish` action is used
- The job has `permissions: id-token: write`
- The PyPI project has GitHub Actions enabled (go to PyPI project → Settings → Trusted Publishers)

### "Package already exists" on Test PyPI

Test PyPI doesn't allow re-uploading the same version. Either:
- Increment the version number
- Use `skip-existing: true` (already configured in workflow)

### Authentication failures

Check that:
- GitHub Secrets are correctly set
- Environment names match (`test-pypi`, `pypi`)
- The API token has appropriate permissions

### Smoke tests fail after installation

Verify that:
- All dependencies are correctly specified in `pyproject.toml`
- The package includes all necessary files (check `MANIFEST.in`)
- Entry points are correctly configured

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)