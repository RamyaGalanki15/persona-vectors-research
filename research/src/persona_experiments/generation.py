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
    """

    chat_prompt: ChatPrompt
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    generated_token_ids: torch.Tensor
    response_text: str


def generate_response(
    model: torch.nn.Module,
    tokenizer: Any,
    system_instruction: str,
    user_question: str,
    generation_config: dict[str, Any],
) -> GenerationResult:
    """
    Generate one assistant response from a configured chat prompt.
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

    input_ids = tokenized_prompt["input_ids"]
    attention_mask = tokenized_prompt["attention_mask"]

    try:
        model_device = next(model.parameters()).device
    except StopIteration as error:
        raise ValueError(
            "The model has no parameters, so its device cannot be determined."
        ) from error

    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    generation_arguments = {
        "max_new_tokens": generation_config["max_new_tokens"],
        "do_sample": generation_config["do_sample"],
        "repetition_penalty": generation_config.get(
            "repetition_penalty",
            1.0,
        ),
    }

    if generation_arguments["do_sample"]:
        generation_arguments["temperature"] = generation_config.get(
            "temperature",
            1.0,
        )
        generation_arguments["top_p"] = generation_config.get(
            "top_p",
            1.0,
        )

    if tokenizer.eos_token_id is not None:
        generation_arguments["eos_token_id"] = tokenizer.eos_token_id

    if tokenizer.pad_token_id is not None:
        generation_arguments["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        generation_arguments["pad_token_id"] = tokenizer.eos_token_id

    model.eval()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_arguments,
        )

    prompt_token_count = input_ids.shape[1]

    generated_token_ids = output_ids[
        :,
        prompt_token_count:,
    ]

    if generated_token_ids.shape[1] == 0:
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

    return GenerationResult(
        chat_prompt=chat_prompt,
        prompt_input_ids=input_ids.detach().cpu(),
        prompt_attention_mask=attention_mask.detach().cpu(),
        generated_token_ids=generated_token_ids.detach().cpu(),
        response_text=response_text,
    )