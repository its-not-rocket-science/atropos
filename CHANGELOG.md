# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.3.0] - 2026-03-19

### Added
- PyPI package release pipeline with automated publishing to PyPI and Test PyPI
- Comprehensive changelog following Keep a Changelog format
- Release preparation scripts (`scripts/prepare_release.py`, `scripts/smoke_test.py`)
- GitHub Actions workflow for PyPI deployment with verification steps
- Project metadata enhancements (URLs, classifiers, maintainer info)
- MANIFEST.in for complete package distribution
- Documentation for maintainers (`docs/pypi-release-guide.md`)

### Changed
- Bumped version from 0.2.0 to 0.3.0 for first PyPI release
- Updated README.md with PyPI installation instructions and badges
- Enhanced pyproject.toml with project URLs and additional classifiers
- Extended Makefile with release targets (`release-test`, `release`, `verify-release`)
- Updated `.github/workflows/release.yml` to include PyPI upload steps

### Fixed
- Ensure version consistency between pyproject.toml and src/atropos/__init__.py

## [0.2.0] - 2026-03-18

### Added
- Wanda pruning compatibility framework integration
- SparseGPT patched framework integration for non-LLaMA models
- Framework-specific strategy presets with quality/speed trade-off analysis
- Comprehensive CI stability improvements and test suite enhancements
- Quantization + pruning combined optimization study with three deliverables
- Git repository cleanup and generated test directory exclusions

### Changed
- Improved CI configuration for better reliability
- Updated ROADMAP.md with completed tasks and active experiments
- Enhanced pruning framework compatibility across diverse model architectures

## [0.1.0] - Initial Release

### Added
- Core Atropos CLI tool for estimating ROI of pruning and quantization optimizations
- Python package with comprehensive feature set:
  - Deployment scenario modeling
  - Optimization strategy composition
  - Calculation engine for energy, cost, CO2e savings
  - Multiple CLI commands: preset, scenario, compare, batch, sensitivity
  - Reporting formats: text, json, markdown, html
  - Web dashboard for interactive exploration
  - Telemetry collection from vLLM/TGI/Triton inference servers
  - Model testing suite for HuggingFace Hub compatibility
  - Pruning framework integrations (LLM-Pruner, Wanda, SparseGPT)
  - Calibration against real performance metrics
  - Atropos Pipeline for automated optimization workflow
- Extensive documentation, tests, CI workflows, and development tooling