# Prompting Refactor Plan for Bundled Environments

## 1) Prompt assumptions tied to model behavior

The hidden-reasoning-era templates generally blend two independent concerns:

1. **Task/evaluation constraints (stable):**
   - Required answer marker (e.g. `#### <number>` for GSM8K).
   - Final output normalization expected by evaluator.
2. **Reasoning-style assumptions (model-dependent):**
   - Instructing the model to emit `<think>...</think>`.
   - Persona prompts like “you are a deep thinking AI”.
   - Provider-specific chain-of-thought phrasing that does not affect scoring.

### Refactor principle
Treat these as separate layers:

- **Benchmark spec layer:** immutable formatting + extraction contract.
- **Reasoning guidance layer:** runtime-selectable hinting policy.

## 2) Proposed prompt abstraction

A compact abstraction was added under `environments/prompting.py`:

- `BenchmarkPromptSpec`
  - `task_instructions`
  - `output_format_requirements`
- `PromptingConfig`
  - `mode`: `no_think | think_tag | provider_reasoning`
  - `provider` (optional)
- `PromptBuilder`
  - Accepts `(spec, config)`
  - Produces `{system, user}` prompt bundle

### Mode semantics

- **no_think**
  - No chain-of-thought request; response constrained to benchmark format.
- **think_tag**
  - Backward-compatible `<think>...</think>` style, but isolated to one mode.
- **provider_reasoning**
  - Provider-specific guidance without hard-coding “deep thinking AI” text.

## 3) Refactor plan for additional environments

1. **Inventory current prompt templates**
   - Locate templates containing `<think>` or “deep thinking AI”.
2. **Extract benchmark contracts**
   - Move answer-format instructions and parse regex into environment-local constants.
3. **Adopt `PromptBuilder`**
   - Replace inline mega-prompt strings with `BenchmarkPromptSpec + PromptingConfig`.
4. **Keep evaluator contracts unchanged**
   - Preserve answer extraction regex + canonical formatting markers.
5. **Default safely**
   - Set default mode to `no_think`; allow opt-in legacy mode where needed.
6. **Regression checks**
   - Compare accuracy and parse-rate before/after by mode.

## 4) Migrated GSM8K example

`environments/gsm8k_server.py` now demonstrates:

- Prompt generation through `PromptBuilder`.
- Stable GSM8K-compatible answer parsing via `ANSWER_PATTERN`.
- Compatibility helper `is_correct` that compares normalized `####` answers.
- `migrated_gsm8k_prompt_example()` showing the same question in all three modes.
