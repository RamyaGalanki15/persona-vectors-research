from types import SimpleNamespace

import torch

from persona_experiments.extraction_pipeline import (
    extract_response_activations,
)


class CharacterTokenizer:
    """
    Maps each character to one token.
    """

    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        input_ids = torch.tensor(
            [[ord(character) for character in text]],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class FakeModel(torch.nn.Module):
    """
    Deterministic model with two transformer layers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

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


def test_pipeline_connects_boundaries_and_hidden_states() -> None:
    """
    Verify that the complete extraction pipeline returns consistent metadata.
    """

    model = FakeModel()
    tokenizer = CharacterTokenizer()

    result = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        prompt_text="abc",
        response_text="de",
        include_embedding_output=False,
    )

    assert result.boundaries.prompt_token_count == 3
    assert result.boundaries.response_token_count == 2

    assert result.hidden_states.prompt_token_count == 3
    assert result.hidden_states.response_token_count == 2

    assert result.hidden_states.transformer_layer_count == 2
    assert result.hidden_states.hidden_size == 2

    expected_means = torch.tensor(
        [
            [13.5, 13.5],
            [23.5, 23.5],
        ]
    )

    assert torch.allclose(
        result.hidden_states.response_token_means,
        expected_means,
    )


def test_pipeline_can_include_embedding_output() -> None:
    """
    Verify that the embedding output can be retained through the pipeline.
    """

    model = FakeModel()
    tokenizer = CharacterTokenizer()

    result = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        prompt_text="ab",
        response_text="cd",
        include_embedding_output=True,
    )

    assert result.hidden_states.transformer_layer_count == 3

    expected_embedding_mean = torch.tensor([2.5, 2.5])

    assert torch.allclose(
        result.hidden_states.response_token_means[0],
        expected_embedding_mean,
    )