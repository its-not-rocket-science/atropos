# Documentation Audit (2026-04-08)
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.


This audit checks inconsistencies across README, docs, configs, and code.

## Inconsistencies

### 1) Project positioning mismatch (README vs package/docs)
- `README.md` frames Atropos primarily as an environment/server/group/trajectory/rollout experimentation and A/B testing system.
- `docs/index.md` and `pyproject.toml` frame Atropos primarily as an ROI estimator for pruning/quantization tradeoffs.
- This causes onboarding confusion about whether Atropos is an experimentation platform first, or an ROI/optimization toolkit first.

**Suggested fix**
- Rewrite `README.md` overview to match the package and docs positioning: ROI estimation + optional pipeline/A-B/validation modules.
- Keep one concise “capabilities map” in README with links to module-specific docs.

**Canonical source of truth**
- `pyproject.toml` (`project.description`) + `docs/index.md` should define product positioning.

### 2) Broken A/B CLI examples in README
- README uses `atropos-llm ab-test create experiment.yaml --start`.
- Actual CLI requires `atropos-llm ab-test create --config experiment.yaml` and has no `--start` flag.
- README sample YAML uses `control_variant` / `treatment_variant` and `metrics` fields, while code expects `variants`, `primary_metric`, and optional `secondary_metrics`.

**Suggested fix**
- Replace README command with:
  - `atropos-llm ab-test create --config experiment.yaml`
- Replace README YAML with the actual `ABTestConfig` shape used by CLI parsing.

**Canonical source of truth**
- `src/atropos/cli.py` argument parser and `src/atropos/abtesting/models.py` data model.

### 3) Invalid Python import paths in docs examples
- `docs/examples.md` and `docs/api.md` use `from atropos-llm import ...`, which is invalid Python syntax.
- Import package name in code is `atropos`; `atropos-llm` is the CLI entry point/package distribution name.

**Suggested fix**
- Replace all Python imports with `from atropos import ...`.
- Add a short doc note: “Use `atropos` in Python, `atropos-llm` in CLI.”

**Canonical source of truth**
- `src/atropos/__init__.py` exports for Python imports.
- `pyproject.toml` `[project.scripts]` for CLI command name.

### 4) Conflicting top-level narrative duplicated across docs
- Core explanation is duplicated in multiple places with different emphasis:
  - README (A/B and trajectory narrative)
  - `docs/index.md` (ROI estimator narrative)
  - `docs/cli.md` (feature-specific usage)
- The same concept (“what Atropos is”) appears in multiple files with divergent framing.

**Suggested fix**
- Keep one canonical “What is Atropos?” narrative in `docs/index.md`.
- Keep README shorter: installation + quickstart + links.
- Remove long conceptual narrative from README unless it exactly mirrors docs/index.

**Canonical source of truth**
- `docs/index.md` for product narrative.
- README only as entry point and navigation.

## Outdated instructions / version mismatches

1. **CLI syntax drift in README A/B flow** (see inconsistency #2).
2. **Python API examples use invalid module name** (see inconsistency #3).
3. **README workflow omits many current first-class commands** (pipeline, validation, telemetry, dashboard, cloud pricing), making it look like A/B testing is the only primary path.

## Duplicate concept detection

### Duplicated concept: “project purpose / system identity”
- Duplicated in `README.md` and `docs/index.md` with different messaging.
- Recommendation: canonical in `docs/index.md`, summary-only in README.

### Duplicated concept: “installation steps”
- Present in README and `docs/installation.md`.
- Recommendation: canonical in `docs/installation.md`; README should link and contain only minimum quickstart.

### Duplicated concept: “CLI usage examples”
- Present in README and `docs/cli.md`.
- Recommendation: canonical command reference in `docs/cli.md`; README keeps one vetted quick command.

## Proposed doc structure (files + responsibilities)

### Top-level
- `README.md`
  - 1-paragraph product summary (aligned with docs/index)
  - ultra-short quickstart (install + one command)
  - links to docs sections
- `CHANGELOG.md`
  - release history only (no usage guides)

### `docs/`
- `index.md` (canonical product narrative)
  - scope, capabilities map, module map
- `installation.md` (canonical install/setup)
  - Python/API install, optional extras, verification
- `cli.md` (canonical CLI reference)
  - generated or manually synced from `argparse` commands
- `api.md` (canonical Python API reference)
  - only valid imports from `atropos`
- `examples.md` (task-oriented examples)
  - runnable snippets validated in CI
- domain docs (keep focused, cross-link heavily)
  - `cloud-cost-modeling.md`
  - `pruning-testing-guide.md`
  - `model-testing-guide.md`
  - `telemetry-collection-guide.md`
  - `dashboard-guide.md`

### Config/schema authority
- `src/atropos/*` dataclasses and parser code remain schema authority.
- `configs/*.yaml` are concrete examples only.
- Add “schema source” callouts in docs pages linking to relevant code modules.

## Suggested follow-up implementation plan

1. Fix invalid imports in `docs/api.md` and `docs/examples.md`.
2. Replace README A/B example command and config format.
3. Add a lightweight doc lint test:
   - fail on `from atropos-llm import`
   - fail on `ab-test create <path>` usage without `--config`
4. Add “Docs ownership map” section in `docs/index.md` to prevent future drift.
