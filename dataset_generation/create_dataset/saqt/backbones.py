from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.hub import load_state_dict_from_url
import torchvision
from torchvision import models


SUPPORTED_RESNETS: Dict[str, Tuple[str, int]] = {
    "resnet18": ("ResNet18_Weights", 512),
    "resnet34": ("ResNet34_Weights", 512),
    "resnet50": ("ResNet50_Weights", 2048),
    "resnet101": ("ResNet101_Weights", 2048),
    "resnet152": ("ResNet152_Weights", 2048),
}


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


@dataclass(frozen=True)
class BackboneBundle:
    model: nn.Module
    transform: torchvision.transforms.Compose
    feature_dim: int
    device: torch.device


def _resolve_weights(model_name: str, weights: str):
    if weights.lower() in {"none", "null"}:
        return None

    weights_enum_name, _ = SUPPORTED_RESNETS[model_name]
    weights_enum = getattr(models, weights_enum_name)
    try:
        return getattr(weights_enum, weights)
    except AttributeError as exc:
        supported = ", ".join(weight.name for weight in weights_enum)
        raise ValueError(
            f"Unsupported weights '{weights}' for {model_name}. Supported weights: {supported}, none"
        ) from exc


def _build_transform(weights_obj) -> torchvision.transforms.Compose:
    if weights_obj is None:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    else:
        transform = weights_obj.transforms()
        mean = getattr(transform, "mean", (0.485, 0.456, 0.406))
        std = getattr(transform, "std", (0.229, 0.224, 0.225))
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=mean, std=std),
        ]
    )


def _load_legacy_resnet_weights(model: nn.Module, model_name: str) -> None:
    state_dict = load_state_dict_from_url(
        model_urls[model_name],
        progress=True,
        map_location="cpu",
    )
    model.load_state_dict(state_dict)


def create_backbone(
    model_name: str = "resnet101",
    weights: str = "DEFAULT",
    device: Optional[str] = None,
) -> BackboneBundle:
    if model_name not in SUPPORTED_RESNETS:
        supported = ", ".join(sorted(SUPPORTED_RESNETS))
        raise ValueError(f"Unsupported model '{model_name}'. Supported models: {supported}")

    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    weights_obj = _resolve_weights(model_name, weights)
    model_func = getattr(models, model_name)
    full_model = model_func(weights=None)
    if weights_obj is not None:
        _load_legacy_resnet_weights(full_model, model_name)
    feature_extractor = nn.Sequential(*list(full_model.children())[:-1])
    feature_extractor.eval().to(resolved_device)

    _, feature_dim = SUPPORTED_RESNETS[model_name]
    return BackboneBundle(
        model=feature_extractor,
        transform=_build_transform(weights_obj),
        feature_dim=feature_dim,
        device=resolved_device,
    )
