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


def validate_path(path_value: str, label: str) -> Path:
    """
    Validate that a required input path exists.
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
    """

    path = Path(path_value)
    path.mkdir(parents=True, exist_ok=True)

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


def execute_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    generation_config: dict[str, Any],
    include_embedding_output: bool,
    results_directory: Path,
) -> list[dict[str, Any]]:
    """
    Generate responses, extract activations, and save sample outputs.
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
            "activation_file": str(activation_path),
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
    print("Experiment execution completed successfully.")


if __name__ == "__main__":
    main()