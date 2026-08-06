from types import SimpleNamespace

import torch

from persona_experiments.hidden_state_extractor import (
    extract_response_hidden_states,
)


class FakeModel(torch.nn.Module):
    """
    Minimal deterministic model for testing hidden-state extraction.

    The model returns:
    - one embedding output;
    - two transformer-layer outputs.

    Every token position has predictable values, allowing the test to verify
    that only response-token positions are averaged.
    """

    def __init__(self) -> None:
        super().__init__()

        # A parameter gives the fake model a device, like a real PyTorch model.
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

        # Each token receives a simple two-dimensional representation:
        #
        # token 0 -> [0, 0]
        # token 1 -> [1, 1]
        # token 2 -> [2, 2]
        # ...
        token_positions = torch.arange(
            sequence_length,
            dtype=torch.float32,
            device=input_ids.device,
        ).view(1, sequence_length, 1)

        base = token_positions.repeat(
            batch_size,
            1,
            hidden_size,
        )

        embedding_output = base
        layer_1_output = base + 10.0
        layer_2_output = base + 20.0

        return SimpleNamespace(
            hidden_states=(
                embedding_output,
                layer_1_output,
                layer_2_output,
            )
        )


def test_extracts_response_token_means_from_transformer_layers() -> None:
    """
    Verify response slicing, embedding exclusion, shapes, means, and norms.
    """

    model = FakeModel()

    # Five tokens total:
    #
    # prompt   -> token positions 0, 1, 2
    # response -> token positions 3, 4
    input_ids = torch.tensor([[101, 102, 103, 104, 105]])
    attention_mask = torch.ones_like(input_ids)

    result = extract_response_hidden_states(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_token_count=3,
        include_embedding_output=False,
    )

    assert result.prompt_token_count == 3
    assert result.response_token_count == 2
    assert result.sequence_token_count == 5

    assert result.response_start_index == 3
    assert result.response_end_index == 5

    # The embedding output is excluded, so only two transformer layers remain.
    assert result.transformer_layer_count == 2
    assert result.hidden_size == 2

    assert result.response_token_means.shape == (2, 2)
    assert result.response_token_norms.shape == (2,)

    # Response positions are 3 and 4, whose mean position is 3.5.
    #
    # Layer 1 adds 10:
    # mean -> [13.5, 13.5]
    #
    # Layer 2 adds 20:
    # mean -> [23.5, 23.5]
    expected_means = torch.tensor(
        [
            [13.5, 13.5],
            [23.5, 23.5],
        ]
    )

    assert torch.allclose(
        result.response_token_means,
        expected_means,
    )

    expected_norms = torch.linalg.vector_norm(
        expected_means,
        dim=1,
    )

    assert torch.allclose(
        result.response_token_norms,
        expected_norms,
    )


def test_can_include_embedding_output() -> None:
    """
    Verify that the embedding output is retained when explicitly requested.
    """

    model = FakeModel()

    input_ids = torch.tensor([[101, 102, 103, 104]])
    attention_mask = torch.ones_like(input_ids)

    result = extract_response_hidden_states(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_token_count=2,
        include_embedding_output=True,
    )

    # Embeddings plus two transformer layers.
    assert result.transformer_layer_count == 3

    # Response positions are 2 and 3, so the positional mean is 2.5.
    expected_embedding_mean = torch.tensor([2.5, 2.5])

    assert torch.allclose(
        result.response_token_means[0],
        expected_embedding_mean,
    )


def test_rejects_sequence_without_response_tokens() -> None:
    """
    Verify that an empty response range raises a clear validation error.
    """

    model = FakeModel()

    input_ids = torch.tensor([[101, 102, 103]])
    attention_mask = torch.ones_like(input_ids)

    try:
        extract_response_hidden_states(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_token_count=3,
        )
    except ValueError as error:
        assert "at least one response token" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for a sequence without response tokens."
        )


def test_rejects_batch_size_greater_than_one() -> None:
    """
    Verify the explicit batch-size-one limitation for Experiment 01.
    """

    model = FakeModel()

    input_ids = torch.tensor(
        [
            [101, 102, 103],
            [201, 202, 203],
        ]
    )
    attention_mask = torch.ones_like(input_ids)

    try:
        extract_response_hidden_states(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_token_count=2,
        )
    except ValueError as error:
        assert "batch_size=1" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for batch size greater than one."
        )