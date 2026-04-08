# Environment API Stability Evaluation and Proposal
> Terminology follows the canonical glossary: `/docs/canonical-glossary.md`.


## Scope
This document evaluates environment-facing APIs in Atropos and proposes a stable, explicit, versioned contract for long-term compatibility.

---

## 1) Current Stability Assessment

### 1.1 Unstable interfaces (today)

1. **Untyped environment variable parsing with implicit coercion**
   - `AtroposConfig.from_env()` reads raw environment variables and immediately casts with `float(...)`.
   - Invalid values fail at runtime with generic exceptions, and there is no schema/version gate on env input format.

2. **Logging env API allows weakly constrained values**
   - Logging env accessors (`ATROPOS_LOG_LEVEL`, `ATROPOS_LOG_FORMAT`, `ATROPOS_LOG_FILE`) perform partial validation and silently downgrade invalid values to defaults via warnings.
   - This creates behavior drift risk for automation that expects hard failures on invalid configuration.

3. **Distributed runtime depends on scheduler/tool-specific env conventions**
   - `init_distributed_pruning()` infers rank/world size from multiple env-key sets (`RANK/WORLD_SIZE`, `SLURM_*`, `OMPI_*`) with priority order encoded in code, not in an explicit contract document.
   - Different launchers can produce incompatible behavior if overlapping keys are present.

4. **Subprocess execution forwards ambient environment by default**
   - Pruning framework command execution copies full `os.environ` and then mutates only a couple keys (`OMP_NUM_THREADS`, `ATROPOS_GPU_MEMORY_GB`).
   - This introduces non-determinism and hidden coupling to host environment state.

### 1.2 Implicit contracts (today)

1. **Key naming conventions are de-facto API**
   - `ATROPOS_*` keys are externally visible but not centrally declared as a versioned, normative contract.

2. **Fallback/precedence behavior is contractual but undocumented**
   - Example: distributed rank selection precedence and logging defaulting semantics are implementation details that callers may accidentally depend on.

3. **Value domains are partially defined in code comments, not machine-enforced**
   - Allowed strings (`text/json`, log levels), ranges, and required/optional semantics are not backed by a strict schema.

4. **Error semantics are inconsistent**
   - Some invalid env values fail fast (e.g., float parse), others warn and continue, and others quietly default.

---

## 2) Proposed Versioned Environment API

## 2.1 Canonical design goals

- **Explicit contract**: one normative spec for all environment keys.
- **Versioned envelope**: clear major/minor compatibility behavior.
- **Strict validation**: deterministic pass/fail with structured errors.
- **Reproducibility**: reduced dependence on ambient process state.
- **Safe evolution**: deprecation windows and compatibility guarantees.

## 2.2 Canonical key space

All stable keys MUST be prefixed with `ATROPOS_ENV__` and carry typed semantics.

Examples:
- `ATROPOS_ENV__VERSION=1.0`
- `ATROPOS_ENV__LOG__LEVEL=INFO`
- `ATROPOS_ENV__RUN__ROOT_SEED=42`
- `ATROPOS_ENV__DIST__BACKEND=nccl`

Legacy keys (`ATROPOS_LOG_LEVEL`, etc.) become compatibility aliases only.

## 2.3 Environment API v1 (clean spec)

```yaml
api:
  name: atropos-environment
  version: 1.0.0
  required:
    - ATROPOS_ENV__VERSION
  properties:
    ATROPOS_ENV__VERSION:
      type: string
      pattern: "^1\\.(0|[1-9]\\d*)$"
      description: "Major/minor env API version expected by runtime"

    ATROPOS_ENV__CONFIG__GRID_CO2E:
      type: number
      minimum: 0.0
      maximum: 5.0
      default: 0.35

    ATROPOS_ENV__CONFIG__HW_SAVINGS_CORR:
      type: number
      minimum: -1.0
      maximum: 1.0
      default: 0.8

    ATROPOS_ENV__REPORT__FORMAT:
      type: string
      enum: [text, markdown, html, json]
      default: text

    ATROPOS_ENV__LOG__LEVEL:
      type: string
      enum: [DEBUG, INFO, WARNING, ERROR, CRITICAL]
      default: WARNING

    ATROPOS_ENV__LOG__FORMAT:
      type: string
      enum: [text, json]
      default: text

    ATROPOS_ENV__LOG__FILE:
      type: string
      format: path
      nullable: true

    ATROPOS_ENV__DIST__ENABLED:
      type: boolean
      default: false

    ATROPOS_ENV__DIST__RANK:
      type: integer
      minimum: 0
      required_if: "ATROPOS_ENV__DIST__ENABLED=true"

    ATROPOS_ENV__DIST__WORLD_SIZE:
      type: integer
      minimum: 1
      required_if: "ATROPOS_ENV__DIST__ENABLED=true"

    ATROPOS_ENV__DIST__LOCAL_RANK:
      type: integer
      minimum: 0
      required_if: "ATROPOS_ENV__DIST__ENABLED=true"

    ATROPOS_ENV__DIST__BACKEND:
      type: string
      enum: [nccl, gloo, mpi]
      default: nccl

    ATROPOS_ENV__DIST__INIT_METHOD:
      type: string
      pattern: "^(env://|tcp://.+|file://.+)$"
      default: env://

    ATROPOS_ENV__EXEC__OMP_NUM_THREADS:
      type: integer
      minimum: 1
      maximum: 512

    ATROPOS_ENV__EXEC__GPU_MEMORY_GB:
      type: number
      minimum: 1
      maximum: 4096

    ATROPOS_ENV__STRICT:
      type: boolean
      default: true
      description: "If true, unknown keys in ATROPOS_ENV__ namespace are fatal"
```

## 2.4 Strict interface definition (runtime model)

```python
@dataclass(frozen=True)
class EnvApiV1:
    version: str
    config_grid_co2e: float = 0.35
    config_hw_savings_corr: float = 0.8
    report_format: Literal["text", "markdown", "html", "json"] = "text"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    log_format: Literal["text", "json"] = "text"
    log_file: Path | None = None
    dist_enabled: bool = False
    dist_rank: int | None = None
    dist_world_size: int | None = None
    dist_local_rank: int | None = None
    dist_backend: Literal["nccl", "gloo", "mpi"] = "nccl"
    dist_init_method: str = "env://"
    exec_omp_num_threads: int | None = None
    exec_gpu_memory_gb: float | None = None
    strict: bool = True
```

Rules:
- Parse once at process start.
- Validate once against versioned schema.
- Produce immutable typed object consumed by all modules.
- No module should read `os.environ` directly after bootstrap.

---

## 3) Validation Layer Proposal

## 3.1 Components

1. **Parser**: maps env keys to typed fields.
2. **Schema validator**: enforces required keys, enums, ranges, and cross-field rules.
3. **Normalizer**: applies defaults and canonical casing.
4. **Diagnostics emitter**: returns structured errors/warnings.

## 3.2 Error model

```json
{
  "code": "ENV_VALIDATION_ERROR",
  "api_version": "1.0",
  "field": "ATROPOS_ENV__DIST__WORLD_SIZE",
  "reason": "required when ATROPOS_ENV__DIST__ENABLED=true",
  "severity": "error",
  "suggestion": "Set a positive integer, e.g. 8"
}
```

- In strict mode: any `error` aborts startup.
- In compatibility mode: legacy alias resolution allowed, but emits deprecation warnings.

## 3.3 Legacy alias map (example)

- `ATROPOS_GRID_CO2E` -> `ATROPOS_ENV__CONFIG__GRID_CO2E`
- `ATROPOS_HW_SAVINGS_CORR` -> `ATROPOS_ENV__CONFIG__HW_SAVINGS_CORR`
- `ATROPOS_REPORT_FORMAT` -> `ATROPOS_ENV__REPORT__FORMAT`
- `ATROPOS_LOG_LEVEL` -> `ATROPOS_ENV__LOG__LEVEL`
- `ATROPOS_LOG_FORMAT` -> `ATROPOS_ENV__LOG__FORMAT`
- `ATROPOS_LOG_FILE` -> `ATROPOS_ENV__LOG__FILE`

---

## 4) Compatibility Guarantees

## 4.1 Guarantees by release type

- **Patch release** (`1.2.3 -> 1.2.4`):
  - No key removals or semantic changes.
  - Validation bug fixes only.

- **Minor release** (`1.2 -> 1.3`):
  - Additive keys only (optional with defaults).
  - Existing keys preserve meaning, precedence, and defaults.

- **Major release** (`1.x -> 2.0`):
  - Breaking changes permitted with migration tooling.
  - Minimum support window: N-1 major via compatibility layer.

## 4.2 Deprecation policy

1. Introduce replacement key in minor version.
2. Keep deprecated key as alias for at least **2 minor releases**.
3. Emit machine-readable deprecation warning with removal target.
4. Remove in next major only.

## 4.3 Reproducibility guarantee

Each run artifact should include:
- effective env API version,
- normalized resolved environment object,
- source-of-truth map (`explicit`, `default`, `legacy-alias`),
- validation diagnostics.

This guarantees replayability and forensic analysis across hosts.

---

## 5) Versioning Strategy

## 5.1 Dual versioning model

1. **Runtime/package version** (e.g., `atropos==0.9.0`) for code releases.
2. **Environment API version** (`ATROPOS_ENV__VERSION`) for config contract stability.

These evolve independently.

## 5.2 Negotiation behavior

At startup:
1. Runtime advertises supported env API range, e.g., `>=1.0,<2.0`.
2. Loader reads `ATROPOS_ENV__VERSION`.
3. If version unsupported:
   - hard fail with actionable message,
   - include nearest supported version.

## 5.3 Practical rollout plan

1. Add `env_api.py` with parser + schema + typed model.
2. Update config/logging/distributed/pruning entry points to consume typed env object.
3. Add compatibility aliases and deprecation warnings.
4. Add contract tests:
   - golden valid env fixtures,
   - invalid fixtures with exact diagnostics,
   - backward compatibility fixtures for legacy keys.
5. Publish migration guide with before/after key mapping.

---

## 6) Implementation Notes for Current Codebase

Priority changes:
1. Centralize environment reads now spread across config, logging, distributed, and subprocess execution paths.
2. Replace ambient subprocess env forwarding with explicit allowlist + explicit overrides.
3. Codify distributed env precedence as documented schema and remove ambiguous inference when strict mode is enabled.
4. Standardize invalid input behavior to a single validation mechanism.
