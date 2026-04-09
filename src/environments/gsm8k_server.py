"""GSM8K benchmark environment with configurable reasoning-style prompting.

This file demonstrates how to preserve GSM8K evaluation compatibility while
removing hard-coded assumptions tied to hidden-reasoning-era prompting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .prompting import (
    BenchmarkPromptSpec,
    PromptBuilder,
    PromptingConfig,
    PromptMode,
)

# GSM8K evaluation expects the canonical answer marker with an integer/string value.
ANSWER_PATTERN = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


@dataclass(frozen=True)
class Gsm8kExample:
    question: str
    answer: str


class Gsm8kEnvironment:
    """Build prompts and extract predictions for GSM8K-compatible evaluation."""

    def __init__(self, prompting: PromptingConfig):
        self._builder = PromptBuilder(
            spec=BenchmarkPromptSpec(
                task_instructions=(
                    "You are solving a grade-school math word problem from GSM8K."
                ),
                output_format_requirements=(
                    "End your response with exactly one line formatted as: #### <number>."
                ),
            ),
            config=prompting,
        )

    def build_prompt(self, example: Gsm8kExample) -> dict[str, str]:
        bundle = self._builder.build(example.question)
        return {"system": bundle.system, "user": bundle.user}

    def parse_prediction(self, model_output: str) -> str | None:
        """Extract normalized numeric prediction from model output."""
        match = ANSWER_PATTERN.search(model_output)
        if not match:
            return None
        return match.group(1).replace(",", "")

    def is_correct(self, example: Gsm8kExample, model_output: str) -> bool:
        predicted = self.parse_prediction(model_output)
        expected = self.parse_prediction(example.answer)
        return predicted is not None and expected is not None and predicted == expected


def migrated_gsm8k_prompt_example() -> dict[str, dict[str, str]]:
    """Illustrate the same benchmark question across supported prompt modes."""
    sample = Gsm8kExample(
        question=(
            "Mia has 3 bags with 4 marbles each. She buys 5 more marbles. "
            "How many marbles now?"
        ),
        answer="#### 17",
    )

    modes = {
        "no_think": Gsm8kEnvironment(
            PromptingConfig(mode=PromptMode.NO_THINK)
        ).build_prompt(sample),
        "think_tag": Gsm8kEnvironment(
            PromptingConfig(mode=PromptMode.THINK_TAG)
        ).build_prompt(sample),
        "provider_reasoning_openai": Gsm8kEnvironment(
            PromptingConfig(mode=PromptMode.PROVIDER_REASONING, provider="openai")
        ).build_prompt(sample),
    }

    return modes
