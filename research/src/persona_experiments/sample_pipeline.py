from dataclasses import dataclass
from typing import Any

import torch

from persona_experiments.extraction_pipeline import (
    ResponseExtractionResult,
    extract_response_activations,
)
from persona_experiments.generation import (
    GenerationResult,
    generate_response,
)


@dataclass(frozen=True)
class SamplePipelineResult:
    """
    Complete result for one generated sample.

    This combines:
    - prompt construction;
    - response generation;
    - response-token boundary validation;
    - layer-wise hidden-state extraction.
    """

    generation: GenerationResult
    extraction: ResponseExtractionResult


def run_sample_pipeline(
    model: torch.nn.Module,
    tokenizer: Any,
    system_instruction: str,
    user_question: str,
    generation_config: dict[str, Any],
    include_embedding_output: bool = False,
) -> SamplePipelineResult:
    """
    Generate one response and extract its response-token activations.
    """

    generation = generate_response(
        model=model,
        tokenizer=tokenizer,
        system_instruction=system_instruction,
        user_question=user_question,
        generation_config=generation_config,
    )

    extraction = extract_response_activations(
        model=model,
        tokenizer=tokenizer,
        prompt_text=generation.chat_prompt.prompt_text,
        response_text=generation.response_text,
        include_embedding_output=include_embedding_output,
    )

    generated_token_count = generation.generated_token_ids.shape[1]

    if generated_token_count != extraction.boundaries.response_token_count:
        raise ValueError(
            "Generated token count does not match the response-token count "
            "obtained by retokenizing the prompt and response text."
        )

    return SamplePipelineResult(
        generation=generation,
        extraction=extraction,
    )