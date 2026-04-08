# Documentation Rewrite Plan

## Goal
Unify documentation around one mental model by centralizing terminology in `/docs/canonical-glossary.md` and removing duplicate concept explanations.

## Canonical source
- **Concept definitions:** `/docs/canonical-glossary.md`
- **Product narrative and navigation:** `docs/index.md`

## Rewrite strategy

1. **Centralize definitions**
   - Keep definitions only in the glossary.
   - Replace repeated in-file definitions with a short glossary reference.

2. **Normalize concept usage**
   - Ensure these terms are used consistently across docs: environment, trajectory, group, rollout, server.
   - Treat “promotion” language as rollout language where applicable.

3. **De-duplicate conceptual intros**
   - Remove long repeated “what is Atropos” sections from non-index docs.
   - Keep local docs focused on workflows, APIs, and operational guidance.

4. **Cross-link instead of re-explaining**
   - Add a standard line near the top of docs that use core terms:
     - “Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.”

5. **Sustainability guardrails**
   - Add docs review check: no alternate definitions for the five core concepts.
   - Require glossary link in new docs that use core concepts.

## Execution phases

### Phase 1 (completed in this change)
- Add canonical glossary.
- Add this rewrite plan.
- Update primary entry docs (`README.md`, `docs/index.md`) to reference glossary.
- Add glossary reference line to docs that discuss core concepts.

### Phase 2
- Tighten wording in legacy deep-dive docs to remove any remaining duplicate concept definitions.
- Consolidate overlapping architecture narratives into `docs/index.md` and keep deep dives implementation-focused.

### Phase 3
- Add doc lint/checks for terminology drift.
- Periodic doc audit to catch reintroduced duplicates.
