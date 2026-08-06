from types import SimpleNamespace

import torch

from persona_experiments.sample_pipeline import run_sample_pipeline


class FakeTokenizer:
    """
    Fake tokenizer with stable prompt/response tokenization.
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

        return "ABC"

    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert add_special_tokens is False

        token_map = {
            "A": 10,
            "B": 11,
            "C": 12,
            "D": 20,
            "E": 21,
        }

        input_ids = torch.tensor(
            [[token_map[character] for character in text]],
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

        reverse_map = {
            20: "D",
            21: "E",
        }

        return "".join(
            reverse_map[token_id]
            for token_id in token_ids.tolist()
        )


class FakeModel(torch.nn.Module):
    """
    Model supporting both generation and hidden-state extraction.
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        continuation = torch.tensor(
            [[20, 21]],
            dtype=torch.long,
            device=input_ids.device,
        )

        return torch.cat(
            [input_ids, continuation],
            dim=1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        batch_size, sequence_length = input_ids.shape
        hidden_size = 2

        positions = torch.arange(
            sequence_length,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, sequence_length, 1)

        base = positions.repeat(
            batch_size,
            1,
            hidden_size,
        )

        return SimpleNamespace(
            hidden_states=(
                base,
                base + 10.0,
                base + 20.0,
            )
        )


def test_runs_generation_and_hidden_state_extraction() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()

    result = run_sample_pipeline(
        model=model,
        tokenizer=tokenizer,
        system_instruction="Instruction",
        user_question="Question",
        generation_config={
            "max_new_tokens": 16,
            "do_sample": False,
            "repetition_penalty": 1.0,
        },
        include_embedding_output=False,
    )

    assert result.generation.response_text == "DE"
    assert result.generation.generated_token_ids.shape == (1, 2)

    assert result.extraction.boundaries.prompt_token_count == 3
    assert result.extraction.boundaries.response_token_count == 2

    assert result.extraction.hidden_states.transformer_layer_count == 2
    assert result.extraction.hidden_states.hidden_size == 2

    expected_means = torch.tensor(
        [
            [13.5, 13.5],
            [23.5, 23.5],
        ]
    )

    assert torch.allclose(
        result.extraction.hidden_states.response_token_means,
        expected_means,
    )