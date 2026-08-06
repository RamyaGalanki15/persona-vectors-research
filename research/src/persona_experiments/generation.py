from dataclasses import dataclass
from typing import Any

import torch

from persona_experiments.chat_prompts import (
    ChatPrompt,
    build_chat_prompt,
)


@dataclass(frozen=True)
class GenerationResult:
    """
    Result from one target-model generation.

    Attributes
    ----------
    chat_prompt:
        Structured chat prompt used for generation.

    prompt_input_ids:
        Token IDs belonging to the prompt.

    prompt_attention_mask:
        Attention mask belonging to the prompt.

    generated_token_ids:
        Newly generated assistant token IDs only.

    response_text:
        Decoded assistant response.

    generated_eos_token:
        Whether the generated continuation contains the tokenizer's EOS token.

    reached_max_new_tokens:
        Whether the generated continuation consumed the configured maximum
        number of new tokens.

    termination_reason:
        Best available termination classification:

        - ``eos_token``
        - ``max_new_tokens``
        - ``other``
    """

    chat_prompt: ChatPrompt
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    generated_token_ids: torch.Tensor
    response_text: str
    generated_eos_token: bool
    reached_max_new_tokens: bool
    termination_reason: str


def get_model_device(
    model: torch.nn.Module,
) -> torch.device:
    """
    Determine the device used by a single-device PyTorch model.
    """

    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError(
            "The model has no parameters, so its device cannot be determined."
        ) from error


def build_generation_arguments(
    tokenizer: Any,
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Build generation arguments from configuration.

    Sampling-only values are disabled explicitly during deterministic
    generation. This prevents Hugging Face from warning that temperature,
    top-p, and top-k settings are being ignored when ``do_sample`` is false.
    """

    max_new_tokens = generation_config["max_new_tokens"]
    do_sample = generation_config["do_sample"]

    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError(
            "generation.max_new_tokens must be a positive integer."
        )

    generation_arguments: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": generation_config.get(
            "repetition_penalty",
            1.0,
        ),
    }

    if do_sample:
        generation_arguments["temperature"] = generation_config.get(
            "temperature",
            1.0,
        )
        generation_arguments["top_p"] = generation_config.get(
            "top_p",
            1.0,
        )

        if "top_k" in generation_config:
            generation_arguments["top_k"] = generation_config["top_k"]

    else:
        # Explicitly override sampling values that may be present in the
        # model's bundled generation configuration.
        generation_arguments["temperature"] = None
        generation_arguments["top_p"] = None
        generation_arguments["top_k"] = None

    eos_token_id = getattr(
        tokenizer,
        "eos_token_id",
        None,
    )

    pad_token_id = getattr(
        tokenizer,
        "pad_token_id",
        None,
    )

    if eos_token_id is not None:
        generation_arguments["eos_token_id"] = eos_token_id

    if pad_token_id is not None:
        generation_arguments["pad_token_id"] = pad_token_id
    elif eos_token_id is not None:
        generation_arguments["pad_token_id"] = eos_token_id

    return generation_arguments


def contains_eos_token(
    generated_token_ids: torch.Tensor,
    eos_token_id: int | list[int] | None,
) -> bool:
    """
    Determine whether generated token IDs contain an EOS token.
    """

    if eos_token_id is None:
        return False

    if isinstance(eos_token_id, int):
        eos_token_ids = [eos_token_id]
    else:
        eos_token_ids = list(eos_token_id)

    return any(
        torch.any(
            generated_token_ids == token_id
        ).item()
        for token_id in eos_token_ids
    )


def determine_termination_reason(
    generated_eos_token: bool,
    reached_max_new_tokens: bool,
) -> str:
    """
    Classify why generation most likely stopped.
    """

    if generated_eos_token:
        return "eos_token"

    if reached_max_new_tokens:
        return "max_new_tokens"

    return "other"


def generate_response(
    model: torch.nn.Module,
    tokenizer: Any,
    system_instruction: str,
    user_question: str,
    generation_config: dict[str, Any],
) -> GenerationResult:
    """
    Generate one assistant response from a configured chat prompt.

    Only newly generated assistant tokens are decoded and returned.
    """

    chat_prompt = build_chat_prompt(
        tokenizer=tokenizer,
        system_instruction=system_instruction,
        user_question=user_question,
    )

    tokenized_prompt = tokenizer(
        chat_prompt.prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    )

    if "input_ids" not in tokenized_prompt:
        raise ValueError(
            "Tokenizer output is missing input_ids."
        )

    if "attention_mask" not in tokenized_prompt:
        raise ValueError(
            "Tokenizer output is missing attention_mask."
        )

    input_ids = tokenized_prompt["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"]

    if input_ids.ndim != 2:
        raise ValueError(
            "Prompt input_ids must have shape "
            "[batch_size, sequence_length]."
        )

    if attention_mask.shape != input_ids.shape:
        raise ValueError(
            "Prompt attention_mask must match input_ids."
        )

    if input_ids.shape[0] != 1:
        raise ValueError(
            "Experiment 01 currently supports batch_size=1."
        )

    model_device = get_model_device(model)

    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    generation_arguments = build_generation_arguments(
        tokenizer=tokenizer,
        generation_config=generation_config,
    )

    model.eval()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_arguments,
        )

    if output_ids.ndim != 2:
        raise ValueError(
            "Generated output IDs must have shape "
            "[batch_size, sequence_length]."
        )

    if output_ids.shape[0] != 1:
        raise ValueError(
            "Generated output must have batch_size=1."
        )

    prompt_token_count = input_ids.shape[1]

    if output_ids.shape[1] < prompt_token_count:
        raise ValueError(
            "Generated output is shorter than the original prompt."
        )

    generated_token_ids = output_ids[
        :,
        prompt_token_count:,
    ]

    generated_token_count = generated_token_ids.shape[1]

    if generated_token_count == 0:
        raise ValueError(
            "The model generated zero response tokens."
        )

    response_text = tokenizer.decode(
        generated_token_ids[0],
        skip_special_tokens=True,
    )

    if not response_text.strip():
        raise ValueError(
            "The decoded model response is empty."
        )

    eos_token_id = generation_arguments.get(
        "eos_token_id"
    )

    generated_eos_token = contains_eos_token(
        generated_token_ids=generated_token_ids,
        eos_token_id=eos_token_id,
    )

    reached_max_new_tokens = (
        generated_token_count
        >= generation_arguments["max_new_tokens"]
    )

    termination_reason = determine_termination_reason(
        generated_eos_token=generated_eos_token,
        reached_max_new_tokens=reached_max_new_tokens,
    )

    return GenerationResult(
        chat_prompt=chat_prompt,
        prompt_input_ids=input_ids.detach().cpu(),
        prompt_attention_mask=attention_mask.detach().cpu(),
        generated_token_ids=generated_token_ids.detach().cpu(),
        response_text=response_text,
        generated_eos_token=generated_eos_token,
        reached_max_new_tokens=reached_max_new_tokens,
        termination_reason=termination_reason,
    )