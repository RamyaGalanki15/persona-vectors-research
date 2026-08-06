from dataclasses import dataclass

import torch

from persona_experiments.hidden_state_extractor import (
    HiddenStateExtraction,
    extract_response_hidden_states,
)
from persona_experiments.token_boundaries import (
    TokenBoundaryResult,
    build_token_boundaries,
)


@dataclass
class ResponseExtractionResult:
    """
    Combined token-boundary and hidden-state extraction result.
    """

    boundaries: TokenBoundaryResult
    hidden_states: HiddenStateExtraction


def get_model_device(model: torch.nn.Module) -> torch.device:
    """
    Determine the device used by a model.

    This helper supports ordinary single-device models. Multi-device loading
    will be handled separately when the real model-loading module is added.
    """

    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError(
            "The model has no parameters, so its device cannot be determined."
        ) from error


def extract_response_activations(
    model: torch.nn.Module,
    tokenizer,
    prompt_text: str,
    response_text: str,
    include_embedding_output: bool = False,
) -> ResponseExtractionResult:
    """
    Validate token boundaries and extract response-token activations.

    Parameters
    ----------
    model:
        Causal language model supporting hidden-state output.

    tokenizer:
        Hugging Face-compatible tokenizer.

    prompt_text:
        Complete text preceding the assistant response.

    response_text:
        Assistant response text.

    include_embedding_output:
        Whether to retain the embedding output in addition to transformer
        layer outputs.

    Returns
    -------
    ResponseExtractionResult
        Validated token boundaries and layer-wise response-token activations.
    """

    boundaries = build_token_boundaries(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        response_text=response_text,
    )

    model_device = get_model_device(model)

    input_ids = boundaries.full_input_ids.to(model_device)
    attention_mask = boundaries.full_attention_mask.to(model_device)

    hidden_states = extract_response_hidden_states(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_token_count=boundaries.prompt_token_count,
        include_embedding_output=include_embedding_output,
    )

    if (
        boundaries.response_token_count
        != hidden_states.response_token_count
    ):
        raise ValueError(
            "Token-boundary and hidden-state response counts do not match."
        )

    return ResponseExtractionResult(
        boundaries=boundaries,
        hidden_states=hidden_states,
    )