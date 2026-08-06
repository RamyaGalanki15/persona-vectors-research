import argparse
import json
from pathlib import Path
from typing import Any

import torch

from persona_experiments.experiment_config import load_experiment_bundle
from persona_experiments.model_loader import load_model_and_tokenizer
from persona_experiments.sample_pipeline import run_sample_pipeline
from persona_experiments.trait_artifact import (
    TraitArtifact,
    load_trait_artifact,
)


DEFAULT_CONFIG_PATH = Path(
    "research/configs/experiments/"
    "experiment_01_hidden_state_extraction.yaml"
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Experiment 01.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute Experiment 01 hidden-state extraction."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Load the model and execute real samples.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=1,
        help="Maximum number of samples to execute.",
    )

    parser.add_argument(
        "--condition",
        choices=["positive", "negative", "both"],
        default="positive",
        help="Contrastive condition to execute.",
    )

    return parser.parse_args()


def validate_path(
    path_value: str,
    label: str,
) -> Path:
    """
    Validate that a required input path exists.

    Parameters
    ----------
    path_value:
        Configured file or directory path.

    label:
        Human-readable path description used in error messages.

    Returns
    -------
    Path
        Validated path.

    Raises
    ------
    FileNotFoundError
        If the configured path does not exist.
    """

    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )

    return path


def ensure_directory(path_value: str) -> Path:
    """
    Create an output directory when it does not already exist.

    Git does not preserve empty directories, so output directories may be
    absent after cloning the repository.
    """

    path = Path(path_value)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def build_execution_samples(
    artifact: TraitArtifact,
    question_count: int,
    positive_instruction_count: int,
    negative_instruction_count: int,
    condition: str,
    max_samples: int,
) -> list[dict[str, Any]]:
    """
    Build a bounded list of positive and/or negative execution samples.

    Samples are ordered deterministically:

    1. positive samples, when requested;
    2. negative samples, when requested;
    3. instruction index;
    4. question index.
    """

    if max_samples <= 0:
        raise ValueError(
            "max_samples must be greater than zero."
        )

    selected_questions = artifact.questions[:question_count]

    samples: list[dict[str, Any]] = []

    if condition in {"positive", "both"}:
        for instruction_index in range(
            positive_instruction_count
        ):
            instruction = artifact.instructions[
                instruction_index
            ].positive

            for question_index, question in enumerate(
                selected_questions
            ):
                samples.append(
                    {
                        "condition": "positive",
                        "instruction_index": instruction_index,
                        "question_index": question_index,
                        "system_instruction": instruction,
                        "user_question": question,
                    }
                )

    if condition in {"negative", "both"}:
        for instruction_index in range(
            negative_instruction_count
        ):
            instruction = artifact.instructions[
                instruction_index
            ].negative

            for question_index, question in enumerate(
                selected_questions
            ):
                samples.append(
                    {
                        "condition": "negative",
                        "instruction_index": instruction_index,
                        "question_index": question_index,
                        "system_instruction": instruction,
                        "user_question": question,
                    }
                )

    return samples[:max_samples]


def compare_repeatability(
    first_result: Any,
    repeated_result: Any,
) -> dict[str, Any]:
    """
    Compare two deterministic executions of the same sample.

    The comparison checks:

    - generated token IDs;
    - decoded response text;
    - activation tensor shape;
    - maximum absolute activation difference;
    - mean absolute activation difference.

    Returns
    -------
    dict[str, Any]
        Repeatability metrics suitable for JSON serialization.

    Raises
    ------
    ValueError
        If activation tensor shapes differ.
    """

    first_tokens = (
        first_result.generation.generated_token_ids
    )

    repeated_tokens = (
        repeated_result.generation.generated_token_ids
    )

    generated_tokens_match = torch.equal(
        first_tokens,
        repeated_tokens,
    )

    response_text_matches = (
        first_result.generation.response_text
        == repeated_result.generation.response_text
    )

    first_activations = (
        first_result.extraction.hidden_states
        .response_token_means
    )

    repeated_activations = (
        repeated_result.extraction.hidden_states
        .response_token_means
    )

    if first_activations.shape != repeated_activations.shape:
        raise ValueError(
            "Repeatability activation shapes do not match. "
            f"First shape: {tuple(first_activations.shape)}. "
            f"Repeated shape: {tuple(repeated_activations.shape)}."
        )

    absolute_difference = torch.abs(
        first_activations
        - repeated_activations
    )

    maximum_absolute_difference = (
        absolute_difference.max().item()
    )

    mean_absolute_difference = (
        absolute_difference.mean().item()
    )

    return {
        "generated_tokens_match": generated_tokens_match,
        "response_text_matches": response_text_matches,
        "activation_shape_matches": True,
        "maximum_absolute_difference": (
            maximum_absolute_difference
        ),
        "mean_absolute_difference": (
            mean_absolute_difference
        ),
        "first_generated_token_count": int(
            first_tokens.shape[1]
        ),
        "repeated_generated_token_count": int(
            repeated_tokens.shape[1]
        ),
    }


def execute_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    generation_config: dict[str, Any],
    include_embedding_output: bool,
    run_repeatability_check: bool,
    results_directory: Path,
) -> list[dict[str, Any]]:
    """
    Generate responses, extract activations, and save sample outputs.

    When repeatability checking is enabled, every sample is executed twice
    with the same deterministic generation settings.
    """

    records: list[dict[str, Any]] = []

    for sample_number, sample in enumerate(
        samples,
        start=1,
    ):
        sample_id = (
            f"sample_{sample_number:03d}_"
            f"{sample['condition']}"
        )

        print()
        print(f"Running {sample_id}")
        print(
            f"Question index: "
            f"{sample['question_index']}"
        )
        print(
            f"Instruction index: "
            f"{sample['instruction_index']}"
        )

        result = run_sample_pipeline(
            model=model,
            tokenizer=tokenizer,
            system_instruction=sample[
                "system_instruction"
            ],
            user_question=sample["user_question"],
            generation_config=generation_config,
            include_embedding_output=(
                include_embedding_output
            ),
        )

        repeatability_result: dict[str, Any] | None = None

        if run_repeatability_check:
            print(
                "Running deterministic repeatability check..."
            )

            repeated_result = run_sample_pipeline(
                model=model,
                tokenizer=tokenizer,
                system_instruction=sample[
                    "system_instruction"
                ],
                user_question=sample["user_question"],
                generation_config=generation_config,
                include_embedding_output=(
                    include_embedding_output
                ),
            )

            repeatability_result = compare_repeatability(
                first_result=result,
                repeated_result=repeated_result,
            )

            print(
                "Generated tokens match: "
                f"{repeatability_result['generated_tokens_match']}"
            )
            print(
                "Response text matches: "
                f"{repeatability_result['response_text_matches']}"
            )
            print(
                "Maximum activation difference: "
                f"{repeatability_result['maximum_absolute_difference']}"
            )
            print(
                "Mean activation difference: "
                f"{repeatability_result['mean_absolute_difference']}"
            )

        hidden_states = result.extraction.hidden_states
        boundaries = result.extraction.boundaries

        activation_filename = (
            f"{sample_id}_response_token_means.pt"
        )

        activation_path = (
            results_directory / activation_filename
        )

        torch.save(
            hidden_states.response_token_means,
            activation_path,
        )

        
        record = {
            "sample_id": sample_id,
            "condition": sample["condition"],
            "instruction_index": sample[
                "instruction_index"
            ],
            "question_index": sample["question_index"],
            "system_instruction": sample[
                "system_instruction"
            ],
            "user_question": sample["user_question"],
            "response_text": (
                result.generation.response_text
            ),
            "prompt_token_count": (
                boundaries.prompt_token_count
            ),
            "response_token_count": (
                boundaries.response_token_count
            ),
            "sequence_token_count": (
                hidden_states.sequence_token_count
            ),
            "response_start_index": (
                boundaries.response_start_index
            ),
            "response_end_index": (
                boundaries.response_end_index
            ),
            "retained_hidden_state_count": (
                hidden_states.transformer_layer_count
            ),
            "hidden_size": hidden_states.hidden_size,
            "include_embedding_output": (
                include_embedding_output
            ),
            "response_token_mean_shape": list(
                hidden_states.response_token_means.shape
            ),
            "response_token_norms": (
                hidden_states.response_token_norms.tolist()
            ),
            "activation_dtype": str(
                hidden_states.response_token_means.dtype
            ),
            "activation_contains_nan": bool(
                torch.isnan(
                    hidden_states.response_token_means
                ).any().item()
            ),
            "activation_contains_inf": bool(
                torch.isinf(
                    hidden_states.response_token_means
                ).any().item()
            ),
            "generation_max_new_tokens": (
                generation_config["max_new_tokens"]
            ),
            "generation_generated_eos_token": (
                result.generation.generated_eos_token
            ),
            "generation_reached_token_limit": (
                result.generation.reached_max_new_tokens
            ),
            "generation_termination_reason": (
                result.generation.termination_reason
            ),
            "activation_file": str(activation_path),
            "repeatability": repeatability_result,
        }

        records.append(record)

        print(
            f"Response tokens: "
            f"{record['response_token_count']}"
        )
        print(
            f"Hidden-state shape: "
            f"{record['response_token_mean_shape']}"
        )
        print(
            "Generation reached token limit: "
            f"{record['generation_reached_token_limit']}"
        )
        print(
            "Generated EOS token: "
            f"{record['generation_generated_eos_token']}"
        )
        print(
            "Generation reached token limit: "
            f"{record['generation_reached_token_limit']}"
        )
        print(
            "Generation termination reason: "
            f"{record['generation_termination_reason']}"
        )
        print(f"Response: {record['response_text']}")

    return records


def save_records(
    records: list[dict[str, Any]],
    results_directory: Path,
    records_filename: str,
    summary_filename: str,
) -> None:
    """
    Save JSONL sample records and a compact experiment summary.
    """

    records_path = (
        results_directory / records_filename
    )

    with records_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    repeatability_results = [
        record["repeatability"]
        for record in records
        if record["repeatability"] is not None
    ]

    summary = {
        "samples_completed": len(records),
        "conditions": [
            record["condition"]
            for record in records
        ],
        "response_token_counts": [
            record["response_token_count"]
            for record in records
        ],
        "hidden_sizes": [
            record["hidden_size"]
            for record in records
        ],
        "retained_hidden_state_counts": [
            record["retained_hidden_state_count"]
            for record in records
        ],
        "activation_shapes": [
            record["response_token_mean_shape"]
            for record in records
        ],
        "generation_reached_token_limit": [
            record["generation_reached_token_limit"]
            for record in records
        ],
        "activation_contains_nan": [
            record["activation_contains_nan"]
            for record in records
        ],
        "activation_contains_inf": [
            record["activation_contains_inf"]
            for record in records
        ],
        "repeatability_checks_completed": len(
            repeatability_results
        ),
        "all_generated_tokens_match": (
            all(
                result["generated_tokens_match"]
                for result in repeatability_results
            )
            if repeatability_results
            else None
        ),
        "all_response_texts_match": (
            all(
                result["response_text_matches"]
                for result in repeatability_results
            )
            if repeatability_results
            else None
        ),
        "maximum_repeatability_difference": (
            max(
                result[
                    "maximum_absolute_difference"
                ]
                for result in repeatability_results
            )
            if repeatability_results
            else None
        ),
        "mean_repeatability_differences": [
            result["mean_absolute_difference"]
            for result in repeatability_results
        ],
        "records_file": str(records_path),
    }

    summary_path = (
        results_directory / summary_filename
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(f"Records saved to: {records_path}")
    print(f"Summary saved to: {summary_path}")


def main() -> None:
    """
    Validate or execute Experiment 01.
    """

    args = parse_args()

    bundle = load_experiment_bundle(
        DEFAULT_CONFIG_PATH
    )

    experiment_config = bundle["experiment"]
    model_config = bundle["model"]
    trait_config = bundle["trait"]

    experiment = experiment_config["experiment"]
    model_settings = model_config["model"]
    generation_config = model_config["generation"]
    trait = trait_config["trait"]
    data = trait_config["data"]
    outputs = experiment_config["outputs"]
    sampling = experiment_config["sampling"]
    activation_config = experiment_config[
        "activation_extraction"
    ]
    validation_config = experiment_config[
        "validation"
    ]

    extraction_artifact = validate_path(
        data["extraction_artifact_path"],
        "Extraction artifact",
    )

    artifact = load_trait_artifact(
        extraction_artifact
    )

    question_count = sampling[
        "number_of_questions"
    ]

    positive_instruction_count = sampling[
        "number_of_positive_instructions"
    ]

    negative_instruction_count = sampling[
        "number_of_negative_instructions"
    ]

    if question_count <= 0:
        raise ValueError(
            "number_of_questions must be greater than zero."
        )

    if positive_instruction_count <= 0:
        raise ValueError(
            "number_of_positive_instructions must be greater than zero."
        )

    if negative_instruction_count <= 0:
        raise ValueError(
            "number_of_negative_instructions must be greater than zero."
        )

    if question_count > len(artifact.questions):
        raise ValueError(
            "Requested more questions than the artifact contains."
        )

    if positive_instruction_count > len(
        artifact.instructions
    ):
        raise ValueError(
            "Requested more positive instructions than "
            "the artifact contains."
        )

    if negative_instruction_count > len(
        artifact.instructions
    ):
        raise ValueError(
            "Requested more negative instructions than "
            "the artifact contains."
        )

    experiment_directory = ensure_directory(
        outputs["experiment_directory"]
    )

    results_directory = ensure_directory(
        outputs["results_directory"]
    )

    logs_directory = ensure_directory(
        outputs["logs_directory"]
    )

    run_repeatability_check = bool(
        validation_config.get(
            "run_repeatability_check",
            False,
        )
    )

    print(
        "Experiment configuration loaded successfully."
    )
    print()
    print(f"Experiment ID: {experiment['id']}")
    print(f"Experiment title: {experiment['title']}")
    print(f"Model ID: {model_settings['model_id']}")
    print(f"Trait: {trait['name']}")
    print(f"Extraction artifact: {extraction_artifact}")
    print(f"Experiment directory: {experiment_directory}")
    print(f"Results directory: {results_directory}")
    print(f"Logs directory: {logs_directory}")
    print()
    print(
        f"Available instruction pairs: "
        f"{len(artifact.instructions)}"
    )
    print(
        f"Available questions: "
        f"{len(artifact.questions)}"
    )
    print(f"Selected questions: {question_count}")
    print(
        f"Selected positive instructions: "
        f"{positive_instruction_count}"
    )
    print(
        f"Selected negative instructions: "
        f"{negative_instruction_count}"
    )
    print(
        f"Repeatability check enabled: "
        f"{run_repeatability_check}"
    )

    if not args.execute:
        print()
        print("Dry run complete. No model was loaded.")
        return

    samples = build_execution_samples(
        artifact=artifact,
        question_count=question_count,
        positive_instruction_count=(
            positive_instruction_count
        ),
        negative_instruction_count=(
            negative_instruction_count
        ),
        condition=args.condition,
        max_samples=args.max_samples,
    )

    print()
    print("Execution requested.")
    print(f"Condition: {args.condition}")
    print(f"Maximum samples: {args.max_samples}")
    print(f"Samples prepared: {len(samples)}")
    print("Loading model and tokenizer...")

    model, tokenizer = load_model_and_tokenizer(
        model_config
    )

    print("Model and tokenizer loaded.")

    records = execute_samples(
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        generation_config=generation_config,
        include_embedding_output=(
            activation_config[
                "include_embedding_output"
            ]
        ),
        run_repeatability_check=(
            run_repeatability_check
        ),
        results_directory=results_directory,
    )

    save_records(
        records=records,
        results_directory=results_directory,
        records_filename=outputs[
            "records_filename"
        ],
        summary_filename=outputs[
            "summary_filename"
        ],
    )

    print()
    print(
        "Experiment execution completed successfully."
    )


if __name__ == "__main__":
    main()