import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ContrastiveInstructionPair:
    """
    One positive and negative instruction pair for a behavioral trait.
    """

    positive: str
    negative: str


@dataclass(frozen=True)
class TraitArtifact:
    """
    Parsed trait-generation artifact.

    Attributes
    ----------
    instructions:
        Contrastive positive and negative instruction pairs.

    questions:
        Questions used to elicit the trait.

    evaluation_prompt:
        Prompt template used for trait scoring.
    """

    instructions: list[ContrastiveInstructionPair]
    questions: list[str]
    evaluation_prompt: str


def load_trait_artifact(
    artifact_path: str | Path,
) -> TraitArtifact:
    """
    Load and validate a trait artifact JSON file.
    """

    path = Path(artifact_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Trait artifact was not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw_artifact: Any = json.load(file)

    if not isinstance(raw_artifact, dict):
        raise ValueError(
            "Trait artifact root must be a JSON object."
        )

    raw_instructions = raw_artifact.get("instruction")
    raw_questions = raw_artifact.get("questions")
    evaluation_prompt = raw_artifact.get("eval_prompt")

    if not isinstance(raw_instructions, list) or not raw_instructions:
        raise ValueError(
            "Trait artifact must contain a non-empty 'instruction' list."
        )

    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(
            "Trait artifact must contain a non-empty 'questions' list."
        )

    if not isinstance(evaluation_prompt, str) or not evaluation_prompt.strip():
        raise ValueError(
            "Trait artifact must contain a non-empty 'eval_prompt' string."
        )

    instructions: list[ContrastiveInstructionPair] = []

    for index, raw_pair in enumerate(raw_instructions):
        if not isinstance(raw_pair, dict):
            raise ValueError(
                f"Instruction entry {index} must be a JSON object."
            )

        positive = raw_pair.get("pos")
        negative = raw_pair.get("neg")

        if not isinstance(positive, str) or not positive.strip():
            raise ValueError(
                f"Instruction entry {index} has an invalid 'pos' value."
            )

        if not isinstance(negative, str) or not negative.strip():
            raise ValueError(
                f"Instruction entry {index} has an invalid 'neg' value."
            )

        instructions.append(
            ContrastiveInstructionPair(
                positive=positive,
                negative=negative,
            )
        )

    questions: list[str] = []

    for index, question in enumerate(raw_questions):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Question entry {index} must be a non-empty string."
            )

        questions.append(question)

    return TraitArtifact(
        instructions=instructions,
        questions=questions,
        evaluation_prompt=evaluation_prompt,
    )