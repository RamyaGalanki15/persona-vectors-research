from dataclasses import dataclass

import torch


@dataclass
class TokenBoundaryResult:
    """
    Tokenized prompt and full sequence with validated response boundaries.
    """

    prompt_input_ids: torch.Tensor
    full_input_ids: torch.Tensor
    full_attention_mask: torch.Tensor
    prompt_token_count: int
    response_token_count: int
    response_start_index: int
    response_end_index: int


def build_token_boundaries(
    tokenizer,
    prompt_text: str,
    response_text: str,
) -> TokenBoundaryResult:
    """
    Tokenize a prompt and prompt-response sequence and identify response tokens.

    This function intentionally checks that the tokenized prompt is an exact
    prefix of the tokenized full sequence. That guards against silent boundary
    errors caused by separately tokenizing ``prompt_text`` and
    ``prompt_text + response_text``.

    Parameters
    ----------
    tokenizer:
        Hugging Face-compatible tokenizer.

    prompt_text:
        Complete text before the assistant response.

    response_text:
        Generated assistant response text.

    Returns
    -------
    TokenBoundaryResult
        Token tensors and validated response-token boundaries.

    Raises
    ------
    ValueError
        If inputs are empty, tokenization fails, the prompt is not an exact
        prefix, or the response contains no tokens.
    """

    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError("prompt_text must be a non-empty string.")

    if not isinstance(response_text, str) or not response_text:
        raise ValueError("response_text must be a non-empty string.")

    full_text = prompt_text + response_text

    prompt_tokens = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    full_tokens = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    prompt_input_ids = prompt_tokens["input_ids"]
    full_input_ids = full_tokens["input_ids"]
    full_attention_mask = full_tokens["attention_mask"]

    if prompt_input_ids.ndim != 2 or full_input_ids.ndim != 2:
        raise ValueError("Tokenizer output must use batch-first 2D tensors.")

    if prompt_input_ids.shape[0] != 1 or full_input_ids.shape[0] != 1:
        raise ValueError("Experiment 01 currently supports batch_size=1.")

    prompt_token_count = prompt_input_ids.shape[1]
    full_token_count = full_input_ids.shape[1]

    if prompt_token_count == 0:
        raise ValueError("The prompt produced zero tokens.")

    if full_token_count <= prompt_token_count:
        raise ValueError(
            "The full sequence must contain at least one response token."
        )

    full_prompt_prefix = full_input_ids[:, :prompt_token_count]

    if not torch.equal(prompt_input_ids, full_prompt_prefix):
        raise ValueError(
            "Prompt tokenization is not an exact prefix of the full sequence. "
            "The prompt-response join may have changed tokenization at the "
            "boundary."
        )

    response_start_index = prompt_token_count
    response_end_index = full_token_count
    response_token_count = response_end_index - response_start_index

    return TokenBoundaryResult(
        prompt_input_ids=prompt_input_ids,
        full_input_ids=full_input_ids,
        full_attention_mask=full_attention_mask,
        prompt_token_count=prompt_token_count,
        response_token_count=response_token_count,
        response_start_index=response_start_index,
        response_end_index=response_end_index,
    )