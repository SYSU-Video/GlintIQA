from __future__ import annotations

from typing import List, Sequence, Tuple

import timm
import torch
from einops import rearrange
from torch import nn
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import Block

from .resnet_backbone import resnet50


class FeatureHook:
    """Collect ViT block outputs used by the VGFE branch."""

    def __init__(self) -> None:
        self.outputs: List[torch.Tensor] = []

    def __call__(self, module: nn.Module, module_in, module_out) -> None:
        self.outputs.append(module_out)

    def clear(self) -> None:
        self.outputs.clear()


class VGFE(nn.Module):
    """ViT-based Global Feature Extractor."""

    def __init__(
        self,
        arch: str = "vit_small_patch16_224",
        feature_indices: Sequence[int] = (6, 7, 8, 9),
        freeze_patch_embed: bool = True,
        freeze_blocks_until: int = 4,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True)
        self.feature_indices = tuple(feature_indices)
        self.hook = FeatureHook()
        self._hook_handles = []
        self._freeze_backbone(freeze_patch_embed, freeze_blocks_until)
        self._register_hooks()

    def _freeze_backbone(self, freeze_patch_embed: bool, freeze_blocks_until: int) -> None:
        for name, module in self.backbone._modules.items():
            if freeze_patch_embed and name.startswith("patch_embed"):
                for param in module.parameters():
                    param.requires_grad = False
            if name.startswith("blocks"):
                for idx, block in enumerate(module):
                    if idx > freeze_blocks_until:
                        break
                    for param in block.parameters():
                        param.requires_grad = False

    def _register_hooks(self) -> None:
        for layer in self.backbone.modules():
            if isinstance(layer, Block):
                self._hook_handles.append(layer.register_forward_hook(self.hook))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        _ = self.backbone(x)
        if len(self.hook.outputs) <= max(self.feature_indices):
            self.hook.clear()
            raise RuntimeError(
                f"VGFE expected block outputs up to index {max(self.feature_indices)}, "
                f"got {len(self.hook.outputs)}."
            )
        features = tuple(self.hook.outputs[idx][:, 1:] for idx in self.feature_indices)
        self.hook.clear()
        return features


class CLFE(nn.Module):
    """CNN-based Local Feature Extractor."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 64 * 4,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.backbone = resnet50()
        self.patch_embed1 = PatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size // 4,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 8,
            patch_size=patch_size // 8,
            in_chans=in_chans * 2,
            embed_dim=embed_dim,
        )
        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 16,
            patch_size=patch_size // 16,
            in_chans=in_chans * 4,
            embed_dim=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat1, feat2, feat3 = self.backbone(x)
        return self.patch_embed1(feat1), self.patch_embed2(feat2), self.patch_embed3(feat3)


class CWSA(nn.Module):
    """Channel-Wise Self-Attention module."""

    def __init__(self, dim: int, drop: float = 0.5, norm_layer=None) -> None:
        super().__init__()
        self.c_q = nn.Linear(dim, dim)
        self.c_k = nn.Linear(dim, dim)
        self.c_v = nn.Linear(dim, dim)
        self.norm_fact = dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(drop)
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, channels, num_tokens = x.shape
        query = torch.nn.functional.normalize(self.c_q(x), dim=-1)
        key = torch.nn.functional.normalize(self.c_k(x), dim=-1)
        value = self.c_v(x)

        attn = query @ key.transpose(-2, -1) * self.norm_fact
        attn = self.softmax(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch_size, channels, num_tokens)
        x = self.proj_drop(x)
        x = self.norm(x)
        return x + residual


class SIEM(nn.Module):
    """Spatial Interaction Enhancement Module."""

    def __init__(self, embed_dim: int, patches_resolution: int, input_multiplier: int = 2) -> None:
        super().__init__()
        self.patches_resolution = patches_resolution
        self.conv = nn.Conv2d(
            embed_dim * input_multiplier,
            embed_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(
            x,
            "b c (h w) -> b c h w",
            h=self.patches_resolution,
            w=self.patches_resolution,
        )
        x = self.conv(x)
        return rearrange(
            x,
            "b c h w -> b (h w) c",
            h=self.patches_resolution,
            w=self.patches_resolution,
        )


class QualityRegressionHead(nn.Module):
    """MLP quality score predictor."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).flatten()

