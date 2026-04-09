# Atropos Minimal Quickstart (Under 10 Minutes)

This is the **smallest first-run path** for a new user:
- one toy environment run
- one toy trainer walkthrough
- zero external services

## 1) Minimum viable stack

Only install what is required for the first run:

- Python 3.10+
- `pip`
- local clone of this repository

No GPU, Docker, model endpoints, dashboard, or optional extras are required.

## 2) First-run path (copy/paste)

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/minimal/toy_env.py
python examples/minimal/toy_trainer_walkthrough.py
```

Expected results:
- `toy_env.py` prints 3 deterministic transitions and a total reward.
- `toy_trainer_walkthrough.py` prints steps, total reward, normalized score, and trajectory.

## 3) Toy environment walkthrough

File: `examples/minimal/toy_env.py`

What it teaches:
1. `reset()` starts an episode.
2. `step(action_text)` returns a deterministic transition.
3. reward = `len(action_text)`.
4. episode ends after `max_steps`.

This is the minimum shape of an environment loop without framework complexity.

## 4) Toy trainer walkthrough

File: `examples/minimal/toy_trainer_walkthrough.py`

What it teaches:
1. A trainer picks an action from candidate strings.
2. The trainer executes a short loop over the toy environment.
3. It produces a tiny report (`steps`, `total_reward`, `normalized_score`, `trajectory`).

This shows the environment → trainer contract with no extra dependencies.

## 5) Onboarding checklist

Use this checklist for first-time setup:

- [ ] Create and activate `.venv`.
- [ ] Install Atropos with `pip install -e .`.
- [ ] Run `python examples/minimal/toy_env.py` successfully.
- [ ] Run `python examples/minimal/toy_trainer_walkthrough.py` successfully.
- [ ] Confirm you can explain: reset, step, reward, done, trajectory.

If all five are complete, you are ready for non-toy examples.
