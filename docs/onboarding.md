# 10-Minute Onboarding (Golden Path)

This guide is optimized for first-time users who want a working run in under 10 minutes.

## 1) Current setup steps (what exists today)

Before this guide, setup was spread across:

1. `README.md` manual virtualenv + install commands.
2. `docs/installation.md` source install + optional dev extras.
3. `docs/cli.md` command examples for individual workflows.
4. `Makefile` developer-oriented targets (`install`, `dev-full`, `test`, etc.).

That required new users to discover and sequence commands manually.

## 2) Friction removed (what is automated now)

We provide a **golden path** with:

- **Single command setup:** `make quickstart`
- **Single command run:** `make run-golden`
- **Single command sanity check:** `make onboarding-check`

These targets:

- create a local `.venv`
- upgrade `pip`/`wheel`
- install Atropos in editable mode
- execute a prewired local run (no external model endpoint required)

## 3) Golden path (copy/paste)

From the repo root:

```bash
make quickstart
make run-golden
```

Expected outcome:

- setup completes with dependencies installed in `.venv`
- a Markdown optimization report prints to terminal from the built-in `medium-coder` scenario

Optional verification:

```bash
make onboarding-check
```

## 4) Default config for first run

Use this minimal scenario if you want a file-based run instead of presets:

```yaml
name: onboarding-local
parameters_b: 7
memory_gb: 8
throughput_toks_per_sec: 30
power_watts: 220
requests_per_day: 10000
tokens_per_request: 800
electricity_cost_per_kwh: 0.12
annual_hardware_cost_usd: 12000
one_time_project_cost_usd: 15000
```

Save it as `examples/onboarding_local.yaml` and run:

```bash
.venv/bin/atropos-llm scenario examples/onboarding_local.yaml --strategy mild_pruning --report markdown
```

## 5) Minimal environment

Recommended baseline:

- Python 3.10+
- GNU Make
- no GPU required
- no external inference server required for golden path

## 6) Prewired model strategy (mock/local)

For onboarding, the fastest path is to use **prewired presets** (`medium-coder` + `mild_pruning`) because they are deterministic and local.

If you need server-style validation later:

- use `ab-test` with local endpoints
- keep onboarding separate from endpoint bring-up

## 7) Suggested tooling improvements (next)

1. Add `make quickstart-offline` using a lockfile + wheelhouse cache.
2. Add `atropos-llm doctor` to validate Python/deps/system readiness.
3. Split dependencies into `core` vs `ml` extras so onboarding avoids heavyweight installs unless needed.
4. Add a generated `onboarding-report.md` artifact target.
5. Add a CI check that executes `make quickstart && make run-golden` on clean environments.

