from types import SimpleNamespace

import pytest
import torch

from persona_experiments.generation import generate_response


class FakeTokenizer:
    """
    Minimal tokenizer supporting chat templating, tokenization, and decoding.
    """

    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True

        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user>"
            "<assistant>"
        )

    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert add_special_tokens is False

        input_ids = torch.tensor(
            [[10, 11, 12]],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool,
    ) -> str:
        assert skip_special_tokens is True

        values = token_ids.tolist()

        if values == [20, 21]:
            return "Generated response"

        return ""


class FakeModel(torch.nn.Module):
    """
    Deterministic generation model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.last_generate_arguments = None

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        self.last_generate_arguments = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs,
        }

        continuation = torch.tensor(
            [[20, 21]],
            dtype=torch.long,
            device=input_ids.device,
        )

        return torch.cat(
            [input_ids, continuation],
            dim=1,
        )


class EmptyGenerationModel(FakeModel):
    """
    Returns only the original prompt tokens.
    """

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return input_ids


def test_generates_and_decodes_only_new_tokens() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()

    result = generate_response(
        model=model,
        tokenizer=tokenizer,
        system_instruction="Prioritize accuracy.",
        user_question="Do you agree?",
        generation_config={
            "max_new_tokens": 32,
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
        },
    )

    assert result.response_text == "Generated response"

    assert result.prompt_input_ids.shape == (1, 3)
    assert result.generated_token_ids.shape == (1, 2)

    assert torch.equal(
        result.generated_token_ids,
        torch.tensor([[20, 21]]),
    )

    arguments = model.last_generate_arguments

    assert arguments["max_new_tokens"] == 32
    assert arguments["do_sample"] is False
    assert arguments["repetition_penalty"] == 1.0
    assert arguments["eos_token_id"] == 99
    assert arguments["pad_token_id"] == 0

    # Temperature and top_p should not be passed for deterministic generation.
    assert "temperature" not in arguments
    assert "top_p" not in arguments


def test_passes_sampling_arguments_when_enabled() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()

    generate_response(
        model=model,
        tokenizer=tokenizer,
        system_instruction="Instruction",
        user_question="Question",
        generation_config={
            "max_new_tokens": 16,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
        },
    )

    arguments = model.last_generate_arguments

    assert arguments["do_sample"] is True
    assert arguments["temperature"] == 0.7
    assert arguments["top_p"] == 0.9
    assert arguments["repetition_penalty"] == 1.05


def test_rejects_zero_generated_tokens() -> None:
    model = EmptyGenerationModel()
    tokenizer = FakeTokenizer()

    with pytest.raises(
        ValueError,
        match="zero response tokens",
    ):
        generate_response(
            model=model,
            tokenizer=tokenizer,
            system_instruction="Instruction",
            user_question="Question",
            generation_config={
                "max_new_tokens": 16,
                "do_sample": False,
            },
        )