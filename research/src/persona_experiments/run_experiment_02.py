from __future__ import annotations

import argparse
import json
from pathlib import Path

from persona_experiments.experiment_config import (
    load_judge_experiment_bundle,
)
from persona_experiments.judging.gemma_judge import (
    GemmaSycophancyJudge,
)


DEFAULT_CONFIG_PATH = Path(
    "research/configs/experiments/"
    "experiment_02_judge_calibration.yaml"
)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for Experiment 02.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute "
            "Experiment 02 judge calibration."
        )
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Load Gemma and execute "
            "judge inference."
        ),
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help=(
            "Maximum number of calibration "
            "samples to judge."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Show detailed model-loading and "
            "generation diagnostics."
        ),
    )

    return parser.parse_args()


def ensure_directory(
    path_value: str | Path,
) -> Path:
    """
    Create an output directory if needed.
    """

    path = Path(path_value)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def load_calibration_examples(
    dataset_path: str | Path,
) -> list[dict]:
    """
    Load and validate calibration examples
    from a JSONL dataset.
    """

    path = Path(dataset_path)

    if not path.exists():
        raise FileNotFoundError(
            "Calibration dataset not found: "
            f"{path}"
        )

    records: list[dict] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(
                    stripped
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in calibration "
                    f"dataset at line {line_number}."
                ) from exc

            required_fields = {
                "sample_id",
                "question",
                "response",
                "human_score",
                "category",
            }

            missing_fields = (
                required_fields
                - record.keys()
            )

            if missing_fields:
                raise ValueError(
                    f"Line {line_number} "
                    f"is missing fields: "
                    f"{sorted(missing_fields)}"
                )

            human_score = int(
                record["human_score"]
            )

            if not 0 <= human_score <= 100:
                raise ValueError(
                    "human_score must be "
                    "between 0 and 100."
                )

            record["human_score"] = (
                human_score
            )

            records.append(
                record
            )

    if not records:
        raise ValueError(
            "Calibration dataset contains "
            "no usable examples."
        )

    sample_ids = [
        record["sample_id"]
        for record in records
    ]

    if len(sample_ids) != len(
        set(sample_ids)
    ):
        raise ValueError(
            "Duplicate sample IDs found "
            "in calibration dataset."
        )

    return records


def print_experiment_header(
    experiment: dict,
    judge_model_settings: dict,
    total_examples: int,
) -> None:
    """
    Print a concise experiment summary.
    """

    print()
    print("=" * 60)
    print(
        "Experiment 02 — "
        "Sycophancy Judge Calibration"
    )
    print("=" * 60)

    print(
        f"Experiment : "
        f"{experiment['id']}"
    )

    print(
        f"Judge      : "
        f"{judge_model_settings['model_id']}"
    )

    print(
        f"Dataset    : "
        f"{total_examples} examples"
    )

    print("=" * 60)


def main() -> None:
    """
    Validate configuration or execute
    Gemma judge calibration.
    """

    args = parse_args()

    bundle = (
        load_judge_experiment_bundle(
            DEFAULT_CONFIG_PATH
        )
    )

    experiment_config = bundle[
        "experiment"
    ]

    judge_model_config = bundle[
        "judge_model"
    ]

    experiment = experiment_config[
        "experiment"
    ]

    data_config = experiment_config[
        "data"
    ]

    outputs = experiment_config[
        "outputs"
    ]

    judge_model_settings = (
        judge_model_config[
            "model"
        ]
    )

    examples = (
        load_calibration_examples(
            data_config[
                "calibration_dataset"
            ]
        )
    )

    results_directory = (
        ensure_directory(
            outputs[
                "results_directory"
            ]
        )
    )

    ensure_directory(
        outputs[
            "logs_directory"
        ]
    )

    print_experiment_header(
        experiment=experiment,
        judge_model_settings=(
            judge_model_settings
        ),
        total_examples=len(examples),
    )

    # ---------------------------------------------------------
    # Dry-run mode
    # ---------------------------------------------------------

    if not args.execute:

        print()
        print(
            "Dry run successful."
        )

        print(
            "Configuration and dataset "
            "validated."
        )

        print(
            "No judge model was loaded."
        )

        print()

        return

    # ---------------------------------------------------------
    # Execution validation
    # ---------------------------------------------------------

    if args.max_samples <= 0:
        raise ValueError(
            "max_samples must be "
            "greater than zero."
        )

    selected_examples = (
        examples[
            :args.max_samples
        ]
    )

    print()
    print(
        f"Samples    : "
        f"{len(selected_examples)}"
    )

    print(
        "Loading judge model..."
    )

    # Detailed model/HF diagnostics are shown
    # only when --verbose is supplied.
    judge = GemmaSycophancyJudge(
        model_config=judge_model_config,
        verbose=args.verbose,
    )

    print(
        "Judge model loaded."
    )

    print()

    # ---------------------------------------------------------
    # Judge inference
    # ---------------------------------------------------------

    predictions: list[dict] = []

    for index, example in enumerate(
        selected_examples,
        start=1,
    ):

        result = judge.judge(
            question=example[
                "question"
            ],
            response=example[
                "response"
            ],
        )

        human_score = int(
            example[
                "human_score"
            ]
        )

        judge_score = int(
            result.sycophancy_score
        )

        record = {
            **example,

            "judge_model": (
                judge.model_id
            ),

            "judge_score": (
                judge_score
            ),

            "judge_semantic_valid": (
                result.semantic_valid
            ),

            "judge_reasoning": (
                result.reasoning
            ),

            "judge_raw_output": (
                result.raw_output
            ),

            "absolute_error": abs(
                human_score
                - judge_score
            ),
        }

        predictions.append(
            record
        )

        # Compact, presentation-friendly output.
        print(
            f"[{index}/"
            f"{len(selected_examples)}] "
            f"{example['sample_id']} "
            f"| human={human_score:3d} "
            f"| judge={judge_score:3d} "
            f"| valid="
            f"{result.semantic_valid}"
        )

        # Reasoning is useful while debugging,
        # but should not clutter normal runs.
        if args.verbose:
            print(
                "  Category  : "
                f"{example['category']}"
            )

            print(
                "  Reasoning : "
                f"{result.reasoning}"
            )

            print()

    # ---------------------------------------------------------
    # Save predictions
    # ---------------------------------------------------------

    output_path = (
        results_directory
        / outputs[
            "predictions_filename"
        ]
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in predictions:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("-" * 60)

    print(
        f"Completed  : "
        f"{len(predictions)} samples"
    )

    print(
        f"Results    : "
        f"{output_path}"
    )

    print(
        "Status     : SUCCESS"
    )

    print("-" * 60)
    print()


if __name__ == "__main__":
    main()