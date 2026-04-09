from environments.gsm8k_server import Gsm8kEnvironment, Gsm8kExample, migrated_gsm8k_prompt_example
from environments.prompting import PromptMode, PromptingConfig


def test_gsm8k_parser_remains_compatible() -> None:
    env = Gsm8kEnvironment(prompting=PromptingConfig(mode=PromptMode.NO_THINK))
    ex = Gsm8kExample(question="q", answer="#### 1,234")

    assert env.parse_prediction("work\n#### 1234") == "1234"
    assert env.is_correct(ex, "analysis\n#### 1234")


def test_modes_generate_distinct_guidance() -> None:
    prompts = migrated_gsm8k_prompt_example()

    assert "<think>" not in prompts["no_think"]["system"]
    assert "<think>" in prompts["think_tag"]["system"]
    assert "Reason internally" in prompts["provider_reasoning_openai"]["system"]


def test_all_modes_preserve_gsm8k_output_contract() -> None:
    prompts = migrated_gsm8k_prompt_example()

    for bundle in prompts.values():
        assert "#### <number>" in bundle["system"]
