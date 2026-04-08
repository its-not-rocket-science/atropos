# Principal Engineer Adoption Assessment (2026-04-08)

## 1) Top 5 blockers to adoption

1. **Product identity is still fractured at the top of the funnel.**
   - README positions Atropos as an RL-style experimentation/A-B platform, while package metadata and docs position it as an ROI estimator for pruning/quantization.
   - New users cannot tell what "hello world" success should look like.

2. **Onboarding has known copy/paste failures in flagship docs.**
   - README A/B CLI command is not aligned with actual CLI argument shape.
   - Docs have invalid Python import examples using `atropos-llm` as a module.

3. **Operational trust is not production-grade yet (timeouts/retries/circuit behavior).**
   - Pipeline executes external commands without hard timeouts in critical paths.
   - Deployment and telemetry network checks remain mostly single-shot.

4. **Observability is too weak for real platform ownership.**
   - No consistent correlation IDs or structured context across run/deployment/stage boundaries.
   - Incidents will be slower and harder to root-cause than they need to be.

5. **Telemetry fidelity is not reliable enough for decision automation.**
   - Collector/parser behavior includes heuristic fallbacks and silent zero-like defaults in failure/degraded paths.
   - If teams do not trust metrics, they will not trust promotion/rejection outcomes.

## 2) Top 5 sources of technical debt

1. **CLI god-module (`src/atropos/cli.py`) is oversized and mixed-responsibility.**
   - Parser definitions, orchestration logic, formatting, and command behavior are concentrated in one ~2.4k-line file.

2. **Resilience behavior is fragmented by entrypoint.**
   - Batch path has stronger timeout/retry controls than pipeline path; same product, different failure semantics.

3. **Subprocess integration pattern is duplicated and unsafe by default.**
   - Repeated shell command execution patterns lack centralized policy for timeout, stderr bounds, retries, and error typing.

4. **Error contracts are inconsistent and often too generic.**
   - Broad exception handling and plain-string failures make machine-level recovery and UX-level guidance hard.

5. **Docs/runtime contract drift has already materialized.**
   - CLI/docs/schema mismatch indicates there is no enforced contract-testing loop between docs and code.

## 3) Three changes that would 10x usability

1. **Ship one guaranteed-working "golden path" that CI executes on every PR.**
   - `atropos init` (template) + `atropos run` (single command) + deterministic artifact output.
   - Remove ambiguity by giving users exactly one trustworthy first journey.

2. **Enforce a docs-as-contract gate in CI.**
   - Validate README/docs commands against CLI parser and validate YAML examples against data models.
   - Any drift should block merge/release.

3. **Add reliability defaults users never have to think about.**
   - Centralized retry/timeout/circuit policies for HTTP + subprocess execution with structured errors and run IDs.
   - This converts "research script behavior" into "platform behavior".

## 4) What should NOT be changed (core strengths)

1. **Do not abandon the cross-domain objective function (cost + performance + quality + carbon).**
   - This is the differentiator vs one-dimensional benchmark tooling.

2. **Do not collapse module boundaries that already exist.**
   - The current split across pipeline, deployment, telemetry, validation, quality, and A/B testing is the right architecture direction.

3. **Do not drop the CLI-first ergonomics.**
   - The CLI is the fastest adoption wedge for infra/practical users; fix it structurally, don’t replace it with notebook-only workflows.

4. **Do not remove reproducibility and validation assets.**
   - Existing validation suites, reports, and docs are a strong trust signal; they should be tightened and productized, not stripped.

## Brutal bottom line

Atropos is **close to being compelling** but still behaves like a high-potential internal research toolkit in key user journeys. The adoption ceiling is not model science; it is **product coherence + reliability + operator confidence**. Solve those three, and adoption accelerates quickly.
