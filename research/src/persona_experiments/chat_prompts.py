from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatPrompt:
    """
    Structured representation of one target-model chat prompt.
    """

    system_instruction: str
    user_question: str
    messages: list[dict[str, str]]
    prompt_text: str


def build_chat_prompt(
    tokenizer: Any,
    system_instruction: str,
    user_question: str,
) -> ChatPrompt:
    """
    Build a target-model prompt using the tokenizer's chat template.

    Parameters
    ----------
    tokenizer:
        Hugging Face-compatible tokenizer with ``apply_chat_template``.

    system_instruction:
        Positive or negative trait instruction.

    user_question:
        Question used to elicit model behavior.

    Returns
    -------
    ChatPrompt
        Structured messages and the rendered prompt text.

    Raises
    ------
    ValueError
        If either text input is empty or the tokenizer does not provide a chat
        template.
    """

    if not isinstance(system_instruction, str) or not system_instruction.strip():
        raise ValueError(
            "system_instruction must be a non-empty string."
        )

    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError(
            "user_question must be a non-empty string."
        )

    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError(
            "Tokenizer must provide apply_chat_template()."
        )

    messages = [
        {
            "role": "system",
            "content": system_instruction,
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if not isinstance(prompt_text, str) or not prompt_text:
        raise ValueError(
            "Tokenizer returned an empty chat prompt."
        )

    return ChatPrompt(
        system_instruction=system_instruction,
        user_question=user_question,
        messages=messages,
        prompt_text=prompt_text,
    )