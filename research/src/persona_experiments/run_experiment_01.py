import argparse
from pathlib import Path

from persona_experiments.experiment_config import load_experiment_bundle
from persona_experiments.trait_artifact import load_trait_artifact


DEFAULT_CONFIG_PATH = Path(
    "research/configs/experiments/"
    "experiment_01_hidden_state_extraction.yaml"
)


def validate_path(path_value: str, label: str) -> Path:
    """
    Validate that a required input path exists.

    This should be used for files or directories that must already exist,
    such as the trait extraction artifact.
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

    Git does not track empty directories, so output folders such as
    ``results`` and ``logs`` may be missing after cloning the repository.
    """

    path = Path(path_value)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path

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

def main() -> None:
    args=parse_args()
    """
    Load and validate Experiment 01 configuration.

    This runner currently performs configuration and artifact validation only.
    It does not load the language model or generate responses yet.
    """

    bundle = load_experiment_bundle(DEFAULT_CONFIG_PATH)

    experiment_config = bundle["experiment"]
    model_config = bundle["model"]
    trait_config = bundle["trait"]

    experiment = experiment_config["experiment"]
    model = model_config["model"]
    trait = trait_config["trait"]
    data = trait_config["data"]
    outputs = experiment_config["outputs"]
    sampling = experiment_config["sampling"]

    # The extraction artifact is an input and must already exist.
    extraction_artifact = validate_path(
        data["extraction_artifact_path"],
        "Extraction artifact",
    )

    artifact = load_trait_artifact(
        extraction_artifact
    )

    question_count = sampling["number_of_questions"]

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
            "Requested more questions than the trait artifact contains."
        )

    if positive_instruction_count > len(artifact.instructions):
        raise ValueError(
            "Requested more positive instructions than the artifact contains."
        )

    if negative_instruction_count > len(artifact.instructions):
        raise ValueError(
            "Requested more negative instructions than the artifact contains."
        )

    # These are output locations. Create them automatically because Git does
    # not preserve empty directories when the repository is cloned.
    experiment_directory = ensure_directory(
        outputs["experiment_directory"]
    )

    results_directory = ensure_directory(
        outputs["results_directory"]
    )

    logs_directory = ensure_directory(
        outputs["logs_directory"]
    )

    print("Experiment configuration loaded successfully.")
    print()
    print(f"Experiment ID: {experiment['id']}")
    print(f"Experiment title: {experiment['title']}")
    print(f"Model ID: {model['model_id']}")
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
    print(
        f"Selected questions: "
        f"{question_count}"
    )
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

    print()
    print("Execution requested.")
    print(f"Condition: {args.condition}")
    print(f"Maximum samples: {args.max_samples}")

    raise NotImplementedError(
        "Real model execution will be connected in the next step."
    )


if __name__ == "__main__":
    main()