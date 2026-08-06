import torch

from persona_experiments.token_boundaries import build_token_boundaries


class CharacterTokenizer:
    """
    Simple tokenizer that maps each character to one integer token.

    Because each character is tokenized independently, prompt tokens remain an
    exact prefix of the full prompt-response token sequence.
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

        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


class BoundaryChangingTokenizer:
    """
    Fake tokenizer that deliberately changes the prompt boundary.

    The standalone prompt is tokenized differently from the same text when it
    appears at the beginning of the full prompt-response sequence.
    """

    def __call__(
        self,
        text: str,
        return_tensors: str,
        add_special_tokens: bool,
    ) -> dict[str, torch.Tensor]:
        if text == "Hello":
            input_ids = torch.tensor([[1, 2]], dtype=torch.long)
        elif text == "Hello world":
            input_ids = torch.tensor([[1, 9, 3]], dtype=torch.long)
        else:
            input_ids = torch.tensor([[7]], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def test_builds_valid_response_boundaries() -> None:
    """
    Verify prompt and response token counts for a stable tokenizer.
    """

    tokenizer = CharacterTokenizer()

    result = build_token_boundaries(
        tokenizer=tokenizer,
        prompt_text="abc",
        response_text="de",
    )

    assert result.prompt_token_count == 3
    assert result.response_token_count == 2
    assert result.response_start_index == 3
    assert result.response_end_index == 5

    assert result.prompt_input_ids.shape == (1, 3)
    assert result.full_input_ids.shape == (1, 5)
    assert result.full_attention_mask.shape == (1, 5)

    assert torch.equal(
        result.prompt_input_ids,
        result.full_input_ids[:, :3],
    )


def test_rejects_empty_prompt() -> None:
    """
    Verify that empty prompts are rejected.
    """

    tokenizer = CharacterTokenizer()

    try:
        build_token_boundaries(
            tokenizer=tokenizer,
            prompt_text="",
            response_text="response",
        )
    except ValueError as error:
        assert "prompt_text must be a non-empty string" in str(error)
    else:
        raise AssertionError("Expected ValueError for an empty prompt.")


def test_rejects_empty_response() -> None:
    """
    Verify that empty responses are rejected.
    """

    tokenizer = CharacterTokenizer()

    try:
        build_token_boundaries(
            tokenizer=tokenizer,
            prompt_text="prompt",
            response_text="",
        )
    except ValueError as error:
        assert "response_text must be a non-empty string" in str(error)
    else:
        raise AssertionError("Expected ValueError for an empty response.")


def test_detects_tokenization_boundary_mismatch() -> None:
    """
    Verify that changed tokenization at the text join is detected.
    """

    tokenizer = BoundaryChangingTokenizer()

    try:
        build_token_boundaries(
            tokenizer=tokenizer,
            prompt_text="Hello",
            response_text=" world",
        )
    except ValueError as error:
        assert "not an exact prefix" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for a tokenization boundary mismatch."
        )