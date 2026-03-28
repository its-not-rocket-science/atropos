# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed
- Updated ROADMAP.md with strategic themes and phased version planning (v0.6.0-v3.0.0) based on community feedback envisioning AI-native optimization, deep observability, team collaboration, multi-model orchestration, regulatory compliance, edge deployment, and autonomous operations

### Fixed

## [0.5.0] - 2026-03-27

### Added
- Multi‑GPU benchmarking support with `DistributedBenchmarkWrapper` for measuring performance across multiple GPUs
- `MultiGPUScalingAnalyzer` for analyzing scaling efficiency across GPU counts with bottleneck detection
- New CLI command `benchmark-multi-gpu` for scaling analysis with markdown and JSON output
- Extended `DeploymentScenario` with `parallel_strategy` field and multi‑GPU throughput scaling in calculations
- Enhanced `MeasuredMetrics` with multi‑GPU specific fields (scaling efficiency, communication overhead, per‑GPU memory)
- Distributed benchmarking integration in validation pipeline with fallback to single‑GPU mode
- Comprehensive unit tests for distributed benchmarking and scaling analysis

### Changed
- Updated roadmap to reflect completion of multi‑GPU benchmarking support and distributed pruning experiments
- Extended `DistributedConfig` with benchmarking‑specific parameters (batch size per GPU, iterations, scaling efficiency measurement)
- Enhanced validation reports with multi‑GPU metrics and improved formatting

### Fixed
- Windows CLI validation hang fixed by deferring torch imports (previous release)

## [0.4.0] - 2026-03-23

### Changed
- Rename package from 'atropos' to 'atropos-llm' due to PyPI name conflict
- Update CLI program name from 'atropos' to 'atropos-llm'
- Update documentation, scripts, and GitHub Actions workflow for new package name
- Update PyPI release guide with new package name and references

## [0.3.1] - 2026-03-21

### Added

### Changed
- Bumped GitHub Actions dependencies to latest versions:
  - actions/checkout from v4 to v6
  - actions/setup-python from v5 to v6
  - actions/upload-artifact from v4 to v7
- Removed debug print statements from pruning and quality measurement scripts
- Applied consistent code formatting with ruff

### Fixed
- Fixed CI submodule fetch failures by pushing missing commit to wanda repository
- Fixed syntax error in patched_prune.py from debug statement removal
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
- Fixed mypy type errors in pruning_integration.py
- Improved CI workflow: skip integration tests, fix environment variables, suppress Node.js deprecation warnings
- Updated release preparation script for Windows compatibility
- Enhanced smoke test for Windows compatibility (timeout, Unicode symbols)

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