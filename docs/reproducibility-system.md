# Reproducibility System Design

## Scope
This design defines a project-wide reproducibility contract for Atropos runs (CLI and Python API), including Monte Carlo analysis, validation benchmarks, pruning workflows, pipeline execution, and A/B test simulation paths.

---

## 1) Sources of nondeterminism

### A. Sampling / stochastic behavior

Current stochastic paths include:

- Monte Carlo uncertainty simulation using Python `random` with optional seed support.
- Validation fallback/simulated metrics with random variance in synthetic mode.
- A/B test simulation helpers that use random draws.
- Potential framework-level stochastic kernels in PyTorch and NumPy paths.

Risk: repeated runs with same inputs can produce different outputs if seed discipline is incomplete or inconsistent across libraries/processes.

### B. Environment randomness and drift

- Dynamic time-based fields (`now`, `time.time`) embedded in outputs.
- Cloud pricing and external API retrieval that can change between runs.
- Model/artifact resolution by mutable tags (e.g., `main`, `latest`) rather than immutable revisions.
- Dependency drift (Python package versions, CUDA/cuDNN/Torch versions, OS/kernel).
- Hardware drift (GPU type/count/driver differences).

Risk: even with fixed seeds, results drift due to changing runtime context.

### C. Async / scheduling / parallel execution nondeterminism

- Multi-process/distributed validation and benchmark execution order.
- Thread/process scheduling in telemetry collection and distributed utilities.
- Potentially nondeterministic reductions and floating-point ordering in distributed or GPU contexts.

Risk: run ordering and floating-point accumulation can diverge unless constrained and logged.

---

## 2) Proposed reproducibility architecture

## 2.1 Seed control system

### A. Hierarchical seed model

Define one root seed and deterministic derivation for all subcomponents:

- `root_seed` (required in reproducible mode)
- `derived_seed(component_name, stage_name, rank, worker_id)` using a stable hash function
- Component-specific seed assignment at startup

Suggested derivation formula:

- `derived = int(sha256(f"{root_seed}:{component}:{stage}:{rank}:{worker}").hexdigest()[:16], 16) % 2**31`

This avoids cross-component coupling and preserves replayability when new components are added.

### B. Deterministic seeding targets

At run start, seed all relevant RNGs:

- Python `random.seed(...)`
- NumPy `np.random.seed(...)` (or Generator with fixed SeedSequence)
- PyTorch CPU + CUDA RNG (`torch.manual_seed`, `torch.cuda.manual_seed_all`)
- Any framework-specific seeds passed through integration wrappers

### C. Deterministic execution toggles

Add strict mode that enforces deterministic settings where available:

- `torch.use_deterministic_algorithms(True)`
- cuDNN deterministic settings (`benchmark=False`, deterministic=True)
- Optional env vars (e.g., CUBLAS workspace settings) documented per backend

If deterministic mode cannot be guaranteed for a selected operation/device, fail fast in strict mode (do not silently continue).

## 2.2 Config versioning

### A. Versioned run config envelope

Wrap existing config in a versioned envelope:

```yaml
reproducibility:
  schema_version: "1.0"
  mode: "strict"   # strict | best_effort | off
  root_seed: 1337
  lock_dependencies: true
  lock_model_revision: true
  capture_env: "full"  # minimal | full

pipeline:
  ... existing config ...
```

### B. Config fingerprinting

Compute canonical config hash:

- Normalize YAML/JSON (sorted keys, stable float/string formatting)
- Hash with SHA-256
- Store as `config_hash`

All run artifacts reference this hash.

### C. Compatibility policy

- Increment `schema_version` on breaking reproducibility metadata changes.
- Maintain migration adapters for prior minor versions.
- Reject unknown major versions in strict mode.

## 2.3 Run metadata logging

Emit a structured run manifest per execution (`run_manifest.json`) plus stage-level events.

### A. Identity fields

- `run_id` (content-addressed deterministic ID in strict mode)
- `parent_run_id` (for resumed/retried runs)
- `started_at`, `ended_at`, `duration_ms`

### B. Code provenance

- `git_commit_sha`, `git_dirty`, `git_branch`
- `source_tree_hash` (optional; strict mode recommended)
- `package_version` (`atropos.__version__`)

### C. Runtime environment

- Python version, platform, kernel
- Installed package lock snapshot (name/version)
- CPU/GPU model, count, driver, CUDA/cuDNN/Torch versions
- Container image digest (if containerized)

### D. Inputs and external dependencies

- `config_hash` + embedded normalized config
- Model identifiers + immutable revisions/digests
- Input dataset fingerprints (path + size + sha256)
- External API snapshots (pricing payload hash + retrieved timestamp)

### E. Determinism controls

- Root seed + derived seeds by component
- Deterministic flags status and enforcement result
- Parallelism settings (workers, ranks, thread counts)

### F. Outputs and metrics

- Output artifact list with hashes
- Final metrics + stage metrics
- Exit status + error taxonomy + traceback hash (if failed)

---

## 3) Definition of a replayable run

A run is **replayable** if another operator can reproduce outputs within declared tolerance using only stored artifacts and documented tooling.

## Required to store (minimum contract)

1. **Exact code identity**
   - Git commit SHA
   - Dirty-state indicator (must be false for strict replay)

2. **Exact configuration**
   - Full normalized config
   - Config schema version
   - Config hash

3. **Exact randomness state**
   - Root seed
   - Seed derivation algorithm version
   - Derived seeds per component/worker

4. **Exact execution environment**
   - Python + package versions (lock snapshot)
   - OS/kernel
   - Torch/CUDA/cuDNN versions
   - Hardware fingerprint

5. **Exact input references**
   - Model revisions/digests
   - Dataset and file hashes
   - External data snapshots or pinned payload hashes

6. **Exact outputs**
   - Result artifacts + checksums
   - Stage outputs/events

7. **Execution policy**
   - Reproducibility mode (`strict`/`best_effort`)
   - Deterministic flags actually applied
   - Parallelism settings

## Replay acceptance criteria

- **Bitwise mode (strict CPU paths):** artifact hashes must match exactly.
- **Numerical mode (GPU/distributed):** metrics must be within declared epsilon tolerances, and all non-metric artifacts (configs/manifests) must hash-match.

---

## 4) CLI / API changes to enforce reproducibility

## CLI additions

### Global flags

- `--repro-mode {strict,best-effort,off}`
- `--seed <int>` (promote to global, not command-specific)
- `--repro-manifest <path>` (default: `<output_dir>/run_manifest.json`)
- `--config-lock <path>` (optional lock file for env/model/input pinning)
- `--fail-on-nondeterminism` (alias for strict fail-fast)
- `--replay <manifest_or_lock_path>` (execute from recorded manifest)

### Command behavior changes

- In `strict` mode, commands fail if:
  - seed missing,
  - git tree dirty (unless `--allow-dirty`),
  - unpinned model revision,
  - unresolved external inputs without snapshots.
- Write manifest for every command that executes a run (`pipeline`, `validate`, `benchmark`, `monte-carlo`, pruning integration).

## Python API additions

### New objects

- `ReproConfig` dataclass
  - `mode`, `root_seed`, `allow_dirty`, `capture_env_level`, `epsilon_policy`
- `RunManifest` model
  - strongly typed schema and JSON export/import

### New helper APIs

- `set_global_reproducibility(repro: ReproConfig) -> None`
- `derive_seed(component: str, stage: str, rank: int = 0, worker: int = 0) -> int`
- `write_run_manifest(manifest: RunManifest, path: Path) -> None`
- `replay_run(manifest_path: Path) -> ReplayResult`

### Integration points

- Initialize reproducibility context at CLI entrypoint.
- Inject derived seeds into each stage runner and external integration call.
- Attach manifest hooks in pipeline/validation/benchmark runners.

---

## Reproducibility checklist

## Before run

- [ ] Select reproducibility mode (`strict` or `best_effort`).
- [ ] Set root seed.
- [ ] Confirm clean git state (or explicitly allow dirty).
- [ ] Pin model revisions/digests.
- [ ] Resolve and hash input files/datasets.
- [ ] Freeze dependencies or capture lock snapshot.
- [ ] Configure deterministic backend flags.

## During run

- [ ] Derive and apply seeds per component/worker.
- [ ] Record stage start/end + parameters.
- [ ] Persist external API payload snapshots/hashes.
- [ ] Capture parallelism and scheduler-relevant settings.

## After run

- [ ] Write manifest (`run_manifest.json`).
- [ ] Hash all output artifacts.
- [ ] Store failure metadata if run fails.
- [ ] Verify replayability (smoke replay in CI).

---

## Required metadata schema (JSON)

```json
{
  "schema_version": "1.0",
  "run": {
    "run_id": "string",
    "parent_run_id": "string|null",
    "mode": "strict|best_effort|off",
    "started_at": "ISO-8601",
    "ended_at": "ISO-8601",
    "duration_ms": 0,
    "status": "success|failed|partial"
  },
  "code": {
    "git_commit_sha": "string",
    "git_branch": "string",
    "git_dirty": false,
    "source_tree_hash": "sha256:string|null",
    "atropos_version": "string"
  },
  "config": {
    "schema_version": "string",
    "normalized": {},
    "config_hash": "sha256:string"
  },
  "determinism": {
    "root_seed": 1337,
    "seed_derivation_version": "v1",
    "component_seeds": {
      "pipeline.assess.rank0.worker0": 12345
    },
    "torch_deterministic": true,
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "parallelism": {
      "num_workers": 1,
      "world_size": 1,
      "omp_num_threads": 1
    }
  },
  "environment": {
    "python_version": "string",
    "platform": "string",
    "kernel": "string",
    "packages": [
      {"name": "numpy", "version": "x.y.z"}
    ],
    "hardware": {
      "cpu": "string",
      "gpus": ["string"],
      "cuda_version": "string|null",
      "cudnn_version": "string|null",
      "driver_version": "string|null"
    },
    "container": {
      "image": "string|null",
      "digest": "string|null"
    }
  },
  "inputs": {
    "models": [
      {"name": "string", "revision": "string", "digest": "string|null"}
    ],
    "datasets": [
      {"path": "string", "size_bytes": 0, "sha256": "string"}
    ],
    "external_snapshots": [
      {
        "source": "cloud_pricing",
        "retrieved_at": "ISO-8601",
        "payload_hash": "sha256:string",
        "snapshot_path": "string"
      }
    ]
  },
  "stages": [
    {
      "name": "assess",
      "started_at": "ISO-8601",
      "ended_at": "ISO-8601",
      "status": "success|failed|skipped",
      "params_hash": "sha256:string",
      "metrics": {}
    }
  ],
  "outputs": {
    "artifacts": [
      {"path": "string", "sha256": "string", "size_bytes": 0}
    ],
    "final_metrics": {},
    "error": {
      "category": "string|null",
      "message": "string|null",
      "traceback_hash": "sha256:string|null"
    }
  }
}
```

---

## Implementation plan

## Phase 1: Foundation (1 sprint)

1. Add `reproducibility.py` module with:
   - `ReproConfig`, seed derivation, deterministic toggles.
2. Introduce `RunManifest` model + JSON serializer.
3. Add canonical config hashing utility.
4. Add global CLI flags and wire to command context.

Deliverable: reproducibility context can be initialized and seed all libraries consistently.

## Phase 2: Instrumentation (1–2 sprints)

1. Integrate manifest capture in:
   - pipeline runner,
   - validation runner,
   - distributed benchmark,
   - monte-carlo path,
   - pruning integration wrappers.
2. Record external snapshots for cloud pricing and other dynamic data.
3. Add artifact hashing and stage event logging.

Deliverable: every major run command emits `run_manifest.json`.

## Phase 3: Replay + Enforcement (1 sprint)

1. Implement `--replay` command flow from manifest.
2. Add strict-mode guardrails (fail on dirty tree/unpinned revisions/missing seed).
3. Add replay verification utility with epsilon policy.

Deliverable: users can rerun from manifest and validate equivalence.

## Phase 4: CI hardening (ongoing)

1. Add CI job that executes a deterministic smoke test twice and compares manifests/artifacts.
2. Add schema compatibility tests and migration tests.
3. Document operator runbook for incident replay.

Deliverable: reproducibility regressions are detected before release.

---

## Notes on practical limits

- Full bitwise reproducibility may be infeasible across heterogeneous GPUs/drivers.
- The system should distinguish:
  - **strict reproducible** (bitwise where possible),
  - **scientifically reproducible** (within numeric tolerance).
- Manifest must explicitly state which guarantee applies to a given run.
