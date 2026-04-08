# Canonical Glossary

This glossary defines the five core Atropos concepts exactly once. All other docs should reference these definitions instead of redefining terms.

## Environment
A deterministic execution boundary that owns `reset`/`step` state transitions for one experiment run.

## Trajectory
The append-only ordered log of an environment run: initial state, per-step inputs/actions/outputs/metrics, and terminal outcome.

## Group
A labeled cohort of traffic or samples used for comparative analysis (for example, control vs treatment) under one experiment design.

## Rollout
The governed promotion process that moves a candidate configuration from experiment results to production exposure using explicit gates.

## Server
An inference endpoint accessed by the environment to execute model requests and return responses plus runtime telemetry.

## Usage rule
When writing docs, use these terms with the meanings above, and link back here rather than introducing alternate definitions.
