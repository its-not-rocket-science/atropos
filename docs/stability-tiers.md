# Stability Tiers

This document defines Atropos stability tiers, compatibility expectations, and release behavior.

Use these tiers when deciding whether to build production dependencies on a module and when evaluating upgrade risk.

## Tier definitions

### Tier 1 — Platform Core

Tier 1 is the production backbone of Atropos. It includes the API/runtime boundary and stateful execution primitives that are expected to be dependable across normal upgrades.

**Representative modules**
- `src/atroposlib/api/*` (API server + runtime state endpoints)
- `src/atroposlib/api/storage.py` (store layer)
- `src/atroposlib/envs/runtime_controller.py` and `src/atroposlib/envs/worker_runtime.py` (runtime loop)
- `src/atroposlib/envs/transport_client.py` (transport and retry behavior)
- `src/atroposlib/workers/runtime.py` (runtime worker process)

**Compatibility expectations**
- Backward compatibility is expected across minor releases for public Python interfaces and HTTP payload schemas.
- Behavior changes should be additive by default; removals/renames require a documented migration path.
- Breaking changes should only occur in major releases unless a security or severe correctness fix requires an urgent exception.

### Tier 2 — Supported Research Features

Tier 2 includes features used in practical workflows and maintained by core contributors, but with less strict long-term interface stability than Tier 1.

**Representative modules**
- `src/atropos/validation/*`
- `src/atropos/abtesting/*`
- `src/atropos/pipeline/*`
- `src/atropos/quality/*`
- `src/atropos/trajectory/*`

**Compatibility expectations**
- Public interfaces are intended to remain usable across minor releases, but parameter shapes and extension points may evolve.
- Deprecation warnings may precede changes, but not every experimental parameter is guaranteed long-term support.
- Consumers should pin minor versions when integrating deeply with these modules.

### Tier 3 — Experimental / Community Environments

Tier 3 includes incubating environment integrations, community plugins, and prototype surfaces.

**Representative modules**
- `src/environments/*`
- `src/atroposlib/plugins/*`
- example/plugin packages under `examples/plugins/*`

**Compatibility expectations**
- Interfaces may change at any release boundary when needed for iteration speed.
- Features may be renamed, moved, or removed with limited notice.
- Production adopters should vendor/fork or pin exact versions if depending on Tier 3 behavior.

## Repository annotation policy

Atropos labels tier status in three places:

1. **Module docstrings** for major packages/components.
2. **README** for user-facing maturity expectations.
3. **This policy document** as the canonical source of tier definitions.

When a module is promoted or demoted, update all three in the same change.

## How to use tiers in practice

- Building production services: prefer Tier 1 interfaces.
- Building internal research pipelines: Tier 2 is appropriate with version pinning.
- Prototyping integrations or custom environments: Tier 3 is expected, but treat it as unstable.
