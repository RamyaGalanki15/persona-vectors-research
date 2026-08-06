from dataclasses import dataclass

import torch


@dataclass
class HiddenStateExtraction:
    """
    Structured result from assistant response-token hidden-state extraction.
    """

    prompt_token_count: int
    response_token_count: int
    sequence_token_count: int
    transformer_layer_count: int
    hidden_size: int
    response_start_index: int
    response_end_index: int
    response_token_means: torch.Tensor
    response_token_norms: torch.Tensor


def extract_response_hidden_states(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_token_count: int,
    include_embedding_output: bool = False,
) -> HiddenStateExtraction:
    """
    Extract layer-wise mean hidden states for assistant response tokens.

    Parameters
    ----------
    model:
        A causal language model that supports ``output_hidden_states=True``.

    input_ids:
        Token IDs for the complete prompt and assistant response.

        Expected shape:

        ``[batch_size, sequence_length]``

        This initial experiment supports a batch size of one.

    attention_mask:
        Attention mask corresponding to ``input_ids``.

    prompt_token_count:
        Number of tokens belonging to the prompt. Response tokens begin at this
        index.

    include_embedding_output:
        Include ``outputs.hidden_states[0]`` when True. By default, only
        transformer-layer outputs are retained.

    Returns
    -------
    HiddenStateExtraction
        Token-boundary metadata and one response-token mean vector per retained
        hidden-state entry.

    Raises
    ------
    ValueError
        If tensor dimensions, batch size, token boundaries, hidden states, or
        response-token counts are invalid.
    """

    if input_ids.ndim != 2:
        raise ValueError(
            "input_ids must have shape [batch_size, sequence_length]."
        )

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "attention_mask must have the same shape as input_ids."
        )

    batch_size, sequence_token_count = input_ids.shape

    if batch_size != 1:
        raise ValueError(
            "Experiment 01 currently supports only batch_size=1."
        )

    if prompt_token_count <= 0:
        raise ValueError(
            "prompt_token_count must be greater than zero."
        )

    if prompt_token_count >= sequence_token_count:
        raise ValueError(
            "The complete sequence must contain at least one response token."
        )

    response_start_index = prompt_token_count
    response_end_index = sequence_token_count
    response_token_count = response_end_index - response_start_index

    model.eval()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states

    if hidden_states is None:
        raise ValueError(
            "The model did not return hidden states."
        )

    if len(hidden_states) < 2:
        raise ValueError(
            "Expected an embedding output and at least one transformer layer."
        )

    # Hugging Face causal language models normally return:
    #
    # hidden_states[0]  -> embedding output
    # hidden_states[1:] -> transformer-layer outputs
    retained_hidden_states = (
        hidden_states
        if include_embedding_output
        else hidden_states[1:]
    )

    response_token_means: list[torch.Tensor] = []
    response_token_norms: list[torch.Tensor] = []

    expected_hidden_size: int | None = None

    for hidden_state_index, hidden_state in enumerate(
        retained_hidden_states
    ):
        if hidden_state.ndim != 3:
            raise ValueError(
                "Each hidden-state tensor must have shape "
                "[batch_size, sequence_length, hidden_size]."
            )

        if hidden_state.shape[0] != batch_size:
            raise ValueError(
                f"Unexpected batch size at hidden-state index "
                f"{hidden_state_index}."
            )

        if hidden_state.shape[1] != sequence_token_count:
            raise ValueError(
                f"Unexpected sequence length at hidden-state index "
                f"{hidden_state_index}."
            )

        hidden_size = hidden_state.shape[2]

        if expected_hidden_size is None:
            expected_hidden_size = hidden_size
        elif hidden_size != expected_hidden_size:
            raise ValueError(
                "Hidden size changed between hidden-state entries."
            )

        response_hidden_states = hidden_state[
            0,
            response_start_index:response_end_index,
            :,
        ]

        if response_hidden_states.shape[0] != response_token_count:
            raise ValueError(
                f"Response-token extraction failed at hidden-state index "
                f"{hidden_state_index}."
            )

        response_mean = response_hidden_states.mean(dim=0).float().cpu()
        response_norm = torch.linalg.vector_norm(response_mean)

        response_token_means.append(response_mean)
        response_token_norms.append(response_norm)

    if expected_hidden_size is None:
        raise ValueError(
            "No hidden-state entries were retained."
        )

    stacked_means = torch.stack(response_token_means, dim=0)
    stacked_norms = torch.stack(response_token_norms, dim=0)

    return HiddenStateExtraction(
        prompt_token_count=prompt_token_count,
        response_token_count=response_token_count,
        sequence_token_count=sequence_token_count,
        transformer_layer_count=len(retained_hidden_states),
        hidden_size=expected_hidden_size,
        response_start_index=response_start_index,
        response_end_index=response_end_index,
        response_token_means=stacked_means,
        response_token_norms=stacked_norms,
    )