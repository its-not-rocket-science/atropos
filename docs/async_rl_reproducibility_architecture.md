# Async RL Reproducibility Architecture

This document defines a deterministic execution model for an asynchronous RL system, with explicit controls for seed propagation, execution ordering, and full-run replay.

## 1) Nondeterministic Sources

### A. Randomness and sampling
- Python `random` calls (exploration policy, data augmentation, randomized resets).
- NumPy RNG use (buffer sampling, preprocessing transforms).
- Framework RNG streams (PyTorch CPU/GPU RNG for policy/value nets).
- Action sampling from stochastic policies (`Categorical`, `Normal`, epsilon-greedy).

### B. Asynchronous scheduling
- Different interleavings of actor/environment/learner tasks.
- Unstable ordering from queue consumers in multi-worker loops.
- Races caused by unguarded shared mutable state (replay buffers, counters, metrics).
- Timing-based behavior (`time.time()` windows, timeout-driven branches).

### C. Runtime/platform variability
- Non-deterministic GPU kernels or reduction order differences.
- Thread pool nondeterminism (BLAS/OpenMP thread scheduling).
- Differences across Python, CUDA, cuDNN, and driver versions.
- Floating-point accumulation order differences in distributed reductions.

### D. External/environmental inputs
- Current wall-clock time, timezone, locale.
- OS entropy (`os.urandom`) usage not tied to run seed.
- Non-versioned environment datasets/config fetched at runtime.

### E. Logging/serialization side effects
- Unstable map/dict iteration if not canonicalized before hashing/ID generation.
- Event timestamps used to derive IDs.
- Partial event flushes on crashes if no write-ahead log.

---

## 2) Deterministic Execution Controls

## 2.1 Seed propagation

Adopt a **single root seed** and derive all sub-seeds by namespace.

### Required rules
1. `run_seed` is mandatory for every training/eval run.
2. Every component receives a deterministic child seed from `(run_seed, component_path, stream_idx)`.
3. No direct calls to global RNGs (`random.*`, `np.random.*`, `torch.*` default generators) in core execution paths.
4. Every stochastic event logs `rng_stream_id` and `rng_counter_before`.

### Deterministic derivation
Use a stable hash/KDF (e.g., SHA-256) and convert to `uint64`:

```text
child_seed = u64(sha256(f"{run_seed}:{component_path}:{stream_idx}"))
```

Suggested namespaces:
- `env/<env_id>/reset`
- `actor/<actor_id>/policy_sample`
- `learner/minibatch_sample`
- `augmentation/<pipeline_stage>`

### Runtime seed manifest
Persist a manifest at run start:

```json
{
  "run_id": "run_2026_04_09_0001",
  "run_seed": 1337,
  "seed_schema_version": 1,
  "streams": {
    "actor/0/policy_sample": 103284948723,
    "actor/1/policy_sample": 92034823443,
    "env/0/reset": 8230948322,
    "learner/minibatch_sample": 5723904823
  }
}
```

## 2.2 Execution ordering control

Use a **deterministic logical clock** and a **single canonical commit path**.

### Event model
Every produced event carries:
- `tick`: global monotonically increasing logical counter.
- `producer_id`: actor/env/learner identifier.
- `local_seq`: producer-local monotonic sequence.
- `event_type`: e.g., `ACTION_CHOSEN`, `ENV_STEP`, `GRADIENT_APPLIED`.

### Ordering rules
1. Producers emit events with `(producer_id, local_seq)`.
2. A deterministic scheduler merges events using fixed comparator:
   1. `tick`
   2. `event_type_priority` (static table)
   3. `producer_id`
   4. `local_seq`
3. Only one component (the commit log writer) mutates global state.
4. Any asynchronous worker communicates through append-only messages; no direct global mutation.

### Deterministic scheduler modes
- **Strict mode (recommended for reproducibility):**
  - Barrier at each tick.
  - All events for tick `t` resolved before `t+1`.
- **Relaxed deterministic mode:**
  - Bounded lookahead allowed.
  - Still sorted by canonical comparator before commit.

### Framework/runtime toggles
- Pin thread counts (`OMP_NUM_THREADS=1`, etc.) where possible.
- Enable deterministic backend flags in ML framework.
- Fail fast when selected ops are known nondeterministic under strict mode.

---

## 3) Replay Format and Minimal Required Data

## 3.1 Replay format

Use an append-only **Run Event Log** (`.jsonl`), with optional compressed chunking.

### File set
1. `manifest.json` — run metadata and versioned determinism config.
2. `events.jsonl` — ordered event stream (canonical replay source).
3. `artifacts/` — optional checkpoints, metrics snapshots.

### `manifest.json` (required fields)
```json
{
  "format_version": "1.0",
  "run_id": "run_2026_04_09_0001",
  "run_seed": 1337,
  "code_commit": "<git_sha>",
  "config_hash": "sha256:<...>",
  "env_spec_hash": "sha256:<...>",
  "determinism": {
    "strict": true,
    "scheduler": "barrier_tick",
    "seed_schema_version": 1
  }
}
```

### `events.jsonl` row shape
```json
{
  "tick": 184,
  "event_id": "184:ACTION_CHOSEN:actor0:91",
  "event_type": "ACTION_CHOSEN",
  "producer_id": "actor0",
  "local_seq": 91,
  "episode_id": 12,
  "step_id": 45,
  "payload": {
    "obs_hash": "sha256:...",
    "action": 1,
    "logprob": -0.31,
    "rng_stream_id": "actor/0/policy_sample",
    "rng_counter_before": 901
  }
}
```

## 3.2 Minimal data required for full deterministic replay

At minimum, persist:
1. **Identity + versions**
   - `format_version`, `run_id`, `code_commit`.
2. **Determinism contract**
   - root seed, seed schema version, scheduler mode, backend determinism flags.
3. **Exact config and environment definitions**
   - canonicalized config blob/hash.
   - env/task spec hash.
4. **Ordered event stream**
   - canonical sorted events with deterministic IDs.
5. **RNG progression markers**
   - stream ID + counter (or raw RNG state snapshots at checkpoints).
6. **State checkpoints (periodic)**
   - model/optimizer/replay-buffer snapshots + hash.

Without (4) and (5), asynchronous replay typically diverges even with same initial seed.

---

## 4) Full-Run Replay Procedure

## 4.1 Replay algorithm

1. Load `manifest.json`.
2. Validate compatibility:
   - same major `format_version`, matching `code_commit` (or approved compatibility map), matching `config_hash`.
3. Initialize deterministic runtime:
   - set root seed and all child streams.
   - enforce deterministic backend/thread flags.
   - instantiate scheduler in manifest mode.
4. Load checkpoints if available; else boot from initial state.
5. Stream `events.jsonl` in order.
6. For each event:
   - verify monotonic `(tick, event_id)`.
   - verify RNG marker before stochastic operation.
   - apply transition/update through the same commit path used online.
7. At checkpoint boundaries, recompute and compare state hashes.
8. At end, compare terminal hash + key metrics against recorded values.

## 4.2 Pseudocode

```python
def replay_run(manifest, events):
    rt = DeterministicRuntime.from_manifest(manifest)
    state = rt.initialize_state()

    last_tick = -1
    for e in events:
        assert e.tick >= last_tick
        last_tick = e.tick

        rt.verify_rng_marker(e)
        state = rt.apply_event(state, e)  # single canonical commit path

        if e.event_type == "CHECKPOINT":
            assert hash_state(state) == e.payload["state_hash"]

    assert hash_state(state) == manifest["terminal_state_hash"]
    return state
```

---

## 5) Reference architecture (component view)

```text
[Config + run_seed]
        |
        v
[Seed Registry] ---> [Per-component RNG streams]
        |
        v
[Async Producers: actors/env/learner]
        |
        v
[Deterministic Scheduler + Ordering Comparator]
        |
        v
[Single Commit Path (state mutation)]
        |
        +--> [Run Event Log (jsonl)]
        +--> [Periodic Checkpoints + Hashes]
        +--> [Metrics]

Replay:
[Manifest + Event Log + Checkpoints] --> [Deterministic Runtime] --> [State/Metrics verification]
```

## 6) Acceptance criteria for "deterministic"

A run is deterministic if, given identical `manifest + events + checkpoints`:
1. Terminal state hash matches.
2. Episode returns and key metrics match bitwise (or documented tolerance if float backend requires).
3. Per-event IDs and ordering are identical.
4. Replay can fail fast with a precise divergence location (`event_id`) when mismatches occur.

