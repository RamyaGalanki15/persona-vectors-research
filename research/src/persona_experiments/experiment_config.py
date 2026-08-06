from pathlib import Path
from typing import Any

from persona_experiments.config_loader import load_yaml_config


def load_experiment_bundle(
    experiment_config_path: str | Path,
) -> dict[str, dict[str, Any]]:
    """
    Load an experiment configuration together with its referenced model
    and trait configurations.

    Parameters
    ----------
    experiment_config_path:
        Path to the experiment YAML file.

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary containing:

        - experiment
        - model
        - trait
    """

    experiment_path = Path(experiment_config_path)

    experiment_config = load_yaml_config(experiment_path)

    references = experiment_config.get("references")

    if not isinstance(references, dict):
        raise ValueError(
            "Experiment configuration must contain a 'references' mapping."
        )

    model_config_path = references.get("model_config")
    trait_config_path = references.get("trait_config")

    if not model_config_path:
        raise ValueError(
            "Experiment configuration is missing 'references.model_config'."
        )

    if not trait_config_path:
        raise ValueError(
            "Experiment configuration is missing 'references.trait_config'."
        )

    model_config = load_yaml_config(model_config_path)
    trait_config = load_yaml_config(trait_config_path)

    return {
        "experiment": experiment_config,
        "model": model_config,
        "trait": trait_config,
    }