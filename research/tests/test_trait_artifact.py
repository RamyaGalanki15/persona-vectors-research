import json
from pathlib import Path

import pytest

from persona_experiments.trait_artifact import load_trait_artifact


def test_loads_valid_trait_artifact(tmp_path: Path) -> None:
    artifact_path = tmp_path / "trait.json"

    artifact_path.write_text(
        json.dumps(
            {
                "instruction": [
                    {
                        "pos": "Agree with the user.",
                        "neg": "Prioritize accuracy.",
                    }
                ],
                "questions": [
                    "Do you agree with me?",
                    "Is my opinion correct?",
                ],
                "eval_prompt": "Score the response.",
            }
        ),
        encoding="utf-8",
    )

    artifact = load_trait_artifact(artifact_path)

    assert len(artifact.instructions) == 1
    assert artifact.instructions[0].positive == "Agree with the user."
    assert artifact.instructions[0].negative == "Prioritize accuracy."

    assert artifact.questions == [
        "Do you agree with me?",
        "Is my opinion correct?",
    ]

    assert artifact.evaluation_prompt == "Score the response."


def test_rejects_missing_instruction_list(tmp_path: Path) -> None:
    artifact_path = tmp_path / "trait.json"

    artifact_path.write_text(
        json.dumps(
            {
                "questions": ["Question"],
                "eval_prompt": "Judge",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="non-empty 'instruction' list",
    ):
        load_trait_artifact(artifact_path)


def test_rejects_invalid_instruction_pair(tmp_path: Path) -> None:
    artifact_path = tmp_path / "trait.json"

    artifact_path.write_text(
        json.dumps(
            {
                "instruction": [
                    {
                        "pos": "Positive",
                    }
                ],
                "questions": ["Question"],
                "eval_prompt": "Judge",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid 'neg' value",
    ):
        load_trait_artifact(artifact_path)


def test_rejects_empty_question(tmp_path: Path) -> None:
    artifact_path = tmp_path / "trait.json"

    artifact_path.write_text(
        json.dumps(
            {
                "instruction": [
                    {
                        "pos": "Positive",
                        "neg": "Negative",
                    }
                ],
                "questions": [""],
                "eval_prompt": "Judge",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Question entry 0",
    ):
        load_trait_artifact(artifact_path)