from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """
    Convert a configured dtype name into a PyTorch dtype.
    """

    if dtype_name not in TORCH_DTYPES:
        supported = ", ".join(sorted(TORCH_DTYPES))

        raise ValueError(
            f"Unsupported torch dtype '{dtype_name}'. "
            f"Supported values: {supported}."
        )

    return TORCH_DTYPES[dtype_name]


def load_model_and_tokenizer(
    model_config: dict[str, Any],
) -> tuple[Any, Any]:
    """
    Load a causal language model and tokenizer from model configuration.
    """

    model_settings = model_config["model"]
    runtime_settings = model_config["runtime"]

    model_id = model_settings["model_id"]
    trust_remote_code = model_settings.get(
        "trust_remote_code",
        False,
    )

    torch_dtype = resolve_torch_dtype(
        runtime_settings["torch_dtype"]
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=runtime_settings.get("device", "auto"),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=runtime_settings.get(
            "low_cpu_mem_usage",
            True,
        ),
        trust_remote_code=trust_remote_code,
    )

    model.eval()

    return model, tokenizer