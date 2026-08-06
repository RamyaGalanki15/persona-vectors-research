from unittest.mock import MagicMock, patch

import pytest
import torch

from persona_experiments.model_loader import (
    load_model_and_tokenizer,
    resolve_torch_dtype,
)


def test_resolve_torch_dtype() -> None:
    assert resolve_torch_dtype("float32") is torch.float32
    assert resolve_torch_dtype("float16") is torch.float16
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16


def test_rejects_unsupported_dtype() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported torch dtype",
    ):
        resolve_torch_dtype("int8")


@patch(
    "persona_experiments.model_loader."
    "AutoModelForCausalLM.from_pretrained"
)
@patch(
    "persona_experiments.model_loader."
    "AutoTokenizer.from_pretrained"
)
def test_loads_model_and_tokenizer_from_config(
    mock_tokenizer_loader: MagicMock,
    mock_model_loader: MagicMock,
) -> None:
    tokenizer = MagicMock()
    model = MagicMock()

    mock_tokenizer_loader.return_value = tokenizer
    mock_model_loader.return_value = model

    model_config = {
        "model": {
            "model_id": "example/model",
            "trust_remote_code": False,
        },
        "runtime": {
            "device": "auto",
            "torch_dtype": "bfloat16",
            "low_cpu_mem_usage": True,
        },
    }

    loaded_model, loaded_tokenizer = load_model_and_tokenizer(
        model_config
    )

    assert loaded_model is model
    assert loaded_tokenizer is tokenizer

    mock_tokenizer_loader.assert_called_once_with(
        "example/model",
        trust_remote_code=False,
    )

    mock_model_loader.assert_called_once_with(
        "example/model",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    model.eval.assert_called_once_with()