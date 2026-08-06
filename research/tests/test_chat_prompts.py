import pytest

from persona_experiments.chat_prompts import build_chat_prompt


class FakeChatTokenizer:
    """
    Minimal tokenizer exposing a deterministic chat template.
    """

    def apply_chat_template(
        self,
        messages,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True

        return (
            f"<system>{messages[0]['content']}</system>"
            f"<user>{messages[1]['content']}</user>"
            "<assistant>"
        )


def test_builds_chat_prompt() -> None:
    tokenizer = FakeChatTokenizer()

    result = build_chat_prompt(
        tokenizer=tokenizer,
        system_instruction="Prioritize accuracy.",
        user_question="Do you agree with me?",
    )

    assert result.system_instruction == "Prioritize accuracy."
    assert result.user_question == "Do you agree with me?"

    assert result.messages == [
        {
            "role": "system",
            "content": "Prioritize accuracy.",
        },
        {
            "role": "user",
            "content": "Do you agree with me?",
        },
    ]

    assert result.prompt_text == (
        "<system>Prioritize accuracy.</system>"
        "<user>Do you agree with me?</user>"
        "<assistant>"
    )


def test_rejects_empty_system_instruction() -> None:
    tokenizer = FakeChatTokenizer()

    with pytest.raises(
        ValueError,
        match="system_instruction must be a non-empty string",
    ):
        build_chat_prompt(
            tokenizer=tokenizer,
            system_instruction="",
            user_question="Question",
        )


def test_rejects_empty_question() -> None:
    tokenizer = FakeChatTokenizer()

    with pytest.raises(
        ValueError,
        match="user_question must be a non-empty string",
    ):
        build_chat_prompt(
            tokenizer=tokenizer,
            system_instruction="Instruction",
            user_question="",
        )


def test_rejects_tokenizer_without_chat_template() -> None:
    tokenizer = object()

    with pytest.raises(
        ValueError,
        match="apply_chat_template",
    ):
        build_chat_prompt(
            tokenizer=tokenizer,
            system_instruction="Instruction",
            user_question="Question",
        )