from pathlib import Path

from persona_experiments.experiment_config import load_experiment_bundle


DEFAULT_CONFIG_PATH = Path(
    "research/configs/experiments/"
    "experiment_01_hidden_state_extraction.yaml"
)


def validate_path(path_value: str, label: str) -> Path:
    """
    Validate that a configured path exists.
    """

    path = Path(path_value)

    if not path.exists():
        raise FileNotFoundError(
            f"{label} does not exist: {path}"
        )

    return path


def main() -> None:
    """
    Load and validate Experiment 01 configuration.
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

    extraction_artifact = validate_path(
        data["extraction_artifact_path"],
        "Extraction artifact",
    )

    experiment_directory = validate_path(
        outputs["experiment_directory"],
        "Experiment directory",
    )

    results_directory = validate_path(
        outputs["results_directory"],
        "Results directory",
    )

    logs_directory = validate_path(
        outputs["logs_directory"],
        "Logs directory",
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


if __name__ == "__main__":
    main()