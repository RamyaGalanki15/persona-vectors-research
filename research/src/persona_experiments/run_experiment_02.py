
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

    return parser.parse_args()


def ensure_directory(
    path_value: str,
) -> Path:

    path = Path(
        path_value
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def load_calibration_examples(
    dataset_path: str | Path,
) -> list[dict]:

    path = Path(
        dataset_path
    )

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

            record = json.loads(
                stripped
            )

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
                record[
                    "human_score"
                ]
            )

            if not 0 <= human_score <= 100:
                raise ValueError(
                    "human_score must be "
                    "between 0 and 100."
                )

            records.append(
                record
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


def main() -> None:

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

    print(
        "Experiment configuration "
        "loaded successfully."
    )

    print()
    print(
        f"Experiment ID: "
        f"{experiment['id']}"
    )

    print(
        f"Experiment title: "
        f"{experiment['title']}"
    )

    print(
        f"Judge model: "
        f"{judge_model_settings['model_id']}"
    )

    print(
        f"Calibration examples: "
        f"{len(examples)}"
    )

    print(
        f"Results directory: "
        f"{results_directory}"
    )

    if not args.execute:
        print()
        print(
            "Dry run complete. "
            "No judge model was loaded."
        )
        return

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
        "Execution requested."
    )

    print(
        f"Samples selected: "
        f"{len(selected_examples)}"
    )

    print(
        "Loading Gemma judge..."
    )

    judge = GemmaSycophancyJudge(
        judge_model_config
    )

    predictions = []

    for index, example in enumerate(
        selected_examples,
        start=1,
    ):

        print()
        print(
            f"[{index}/"
            f"{len(selected_examples)}] "
            f"{example['sample_id']}"
        )

        result = judge.judge(
            question=example[
                "question"
            ],
            response=example[
                "response"
            ],
        )

        record = {
            **example,

            "judge_model": (
                judge.model_id
            ),

            "judge_score": (
                result
                .sycophancy_score
            ),

            "judge_semantic_valid": (
                result
                .semantic_valid
            ),

            "judge_reasoning": (
                result.reasoning
            ),

            "judge_raw_output": (
                result.raw_output
            ),

            "absolute_error": abs(
                example["human_score"]
                - result
                .sycophancy_score
            ),
        }

        predictions.append(
            record
        )

        print(
            f"Human score: "
            f"{example['human_score']}"
        )

        print(
            f"Gemma score: "
            f"{result.sycophancy_score}"
        )

        print(
            f"Reasoning: "
            f"{result.reasoning}"
        )

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

    print()
    print(
        f"Predictions saved to: "
        f"{output_path}"
    )

    print()
    print(
        "Experiment 02 smoke test "
        "completed successfully."
    )


if __name__ == "__main__":
    main()