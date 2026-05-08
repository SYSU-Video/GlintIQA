from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import torch


LEGACY_KEY_PREFIX_MAP = {
    "transformer.": "vgfe.backbone.",
    "cnn.": "clfe.backbone.",
    "patch_embed1.": "clfe.patch_embed1.",
    "patch_embed2.": "clfe.patch_embed2.",
    "patch_embed3.": "clfe.patch_embed3.",
    "att_cnn.": "local_cwsa.",
    "proj.": "local_projection.",
    "fusion1.": "cwsa1.",
    "fusion2.": "cwsa2.",
    "fusion3.": "cwsa3.",
    "fusion4.": "cwsa4.",
    "conv1.": "siem1.conv.",
    "conv2.": "siem2.conv.",
    "conv3.": "siem3.conv.",
    "conv4.": "siem4.conv.",
    "out.": "quality_head.layers.",
}


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def remap_legacy_glintiqa_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        for legacy_prefix, current_prefix in LEGACY_KEY_PREFIX_MAP.items():
            if new_key.startswith(legacy_prefix):
                new_key = current_prefix + new_key[len(legacy_prefix) :]
                break
        remapped[new_key] = value
    return remapped


def save_checkpoint(
    path: Path,
    model,
    optimizer=None,
    epoch: int = 0,
    best_srocc: float = 0.0,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "state_dict": unwrap_model(model).state_dict(),
        "best_srocc": best_srocc,
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if extra:
        checkpoint.update(dict(extra))
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model, optimizer=None, strict: bool = True):
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
    state_dict = remap_legacy_glintiqa_keys(state_dict)
    unwrap_model(model).load_state_dict(state_dict, strict=strict)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
