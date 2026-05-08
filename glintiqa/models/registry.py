from __future__ import annotations

from typing import Any, Callable, Dict

from .glintiqa import GlintIQA


MODEL_REGISTRY: Dict[str, Callable[..., GlintIQA]] = {
    "glintiqa": GlintIQA,
    "IQAModel": GlintIQA,
}


def create_iqa_model(model_name: str = "glintiqa", **kwargs: Any):
    if model_name not in MODEL_REGISTRY:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")
    return MODEL_REGISTRY[model_name](**kwargs)

