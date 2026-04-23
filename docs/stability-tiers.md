# Stability Tiers

This document defines Atropos module stability tiers, compatibility expectations, and CI/testing policy.

Use these tiers to choose dependencies, review upgrade risk, and decide what quality bar applies to a code path.

## Tier definitions

### Tier 1: Platform core

Tier 1 is the production backbone of Atropos. It includes runtime/API/store contracts that are expected to be dependable across normal upgrades.

**Primary module scope**
- `src/atroposlib/api/*`
- `src/atroposlib/envs/runtime_controller.py`
- `src/atroposlib/envs/worker_runtime.py`
- `src/atroposlib/envs/transport_client.py`
- `src/atroposlib/api/storage.py`
- `src/atroposlib/workers/runtime.py`
- `src/atroposlib/observability.py`

**Compatibility expectations**
- Backward compatibility is expected across minor releases for public Python interfaces and HTTP request/response schemas.
- Behavior changes should be additive by default.
- Removals/renames require deprecation notice and a documented migration path.
- Breaking changes are major-release changes unless required for urgent security/correctness fixes.

### Tier 2: Supported research infrastructure

Tier 2 includes maintained research and decision-support infrastructure that is actively supported, but may evolve faster than the platform core.

**Primary module scope**
- `src/atropos/pipeline/*`
- `src/atropos/abtesting/*`
- `src/atropos/quality/*`
- `src/atropos/trajectory/*`
- `src/atropos/validation/*`
- `src/atropos/telemetry.py`
- `src/atropos/telemetry_collector.py`

**Compatibility expectations**
- Interfaces are intended to remain usable across minor releases.
- Parameter shapes and extension points may evolve when needed.
- Deprecation notices are preferred before high-impact changes.
- Downstream users should pin minor versions for deep integrations.

### Tier 3: Experimental/community code

Tier 3 includes prototype, community, and fast-iteration surfaces.

**Primary module scope**
- `src/environments/*`
- `src/atroposlib/plugins/*`
- `examples/plugins/*`
- `examples/*` (except where explicitly promoted)
- `scripts/*`

**Compatibility expectations**
- Interfaces may change at any release boundary.
- Features may be renamed, moved, or removed with limited notice.
- Production users should vendor/fork or pin exact versions when depending on Tier 3 behavior.

## Compatibility policy by tier

| Area | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Public API/runtime contracts | Strong compatibility target | Best-effort compatibility | No compatibility guarantee |
| Deprecation requirement | Required for planned breaks | Strongly preferred | Optional |
| Migration notes | Required for behavior/schema shifts | Recommended for major UX breaks | Optional |
| Operational hardening expectation | Required | Recommended | Optional |

## Experimental surface marking policy

Atropos marks experimental/community surfaces in the following ways:

1. Module/package docstrings (for major entry-point packages).
2. README and docs maturity maps.
3. This policy page as canonical source.

When promoting or demoting a module, update all three in one change.

## Test and CI expectations by tier

Tiers define minimum expectations for test coverage in CI gates.

- **Tier 1 (platform core)**
  - Must remain covered by default CI unit/contract/API checks.
  - Regressions in Tier 1 tests block merge.
  - Runtime/API schema behavior changes require targeted tests.

- **Tier 2 (supported research infrastructure)**
  - Must be covered by default unit/integration tests where practical.
  - Regressions block merge for maintained modules, with short-lived exceptions only when explicitly documented.
  - Changes should include tests for new public behavior.

- **Tier 3 (experimental/community code)**
  - Smoke tests and focused checks are encouraged, but not all paths are hard-blocking.
  - Breakage may be tolerated temporarily if it does not affect Tier 1/Tier 2 guarantees.

## Working guidance

- Building production services: prefer Tier 1 modules.
- Building internal research workflows: Tier 2 is appropriate with minor-version pinning.
- Prototyping custom integrations: Tier 3 is acceptable, but expect churn.

## Operating mode alignment

- **Research mode:** Tier 2 + Tier 3 are acceptable when optimizing for speed.
- **Local dev mode:** validate Tier 1 contracts and develop surrounding behavior in Tier 2.
- **Production mode:** anchor critical paths on Tier 1; adopt Tier 2 selectively with version pinning and rollout safeguards; avoid Tier 3 in critical paths.
