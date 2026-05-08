from __future__ import annotations

import torch
from torch import nn

from .modules import CLFE, CWSA, QualityRegressionHead, SIEM, VGFE


class GlintIQA(nn.Module):
    """Global-Local progressive Integration model for NR-IQA."""

    def __init__(
        self,
        arch: str = "vit_small_patch16_224",
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 64 * 4,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.vgfe = VGFE(arch=arch)
        self.clfe = CLFE(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.patches_resolution = img_size // patch_size

        self.local_cwsa = CWSA(dim=self.patches_resolution**2, norm_layer=nn.LayerNorm)
        self.local_projection = nn.Linear(embed_dim * 3, embed_dim)

        self.cwsa1 = CWSA(dim=self.patches_resolution**2, norm_layer=nn.LayerNorm)
        self.cwsa2 = CWSA(dim=self.patches_resolution**2, norm_layer=nn.LayerNorm)
        self.cwsa3 = CWSA(dim=self.patches_resolution**2, norm_layer=nn.LayerNorm)
        self.cwsa4 = CWSA(dim=self.patches_resolution**2, norm_layer=nn.LayerNorm)
        self.siem1 = SIEM(embed_dim=embed_dim, patches_resolution=self.patches_resolution)
        self.siem2 = SIEM(embed_dim=embed_dim, patches_resolution=self.patches_resolution)
        self.siem3 = SIEM(embed_dim=embed_dim, patches_resolution=self.patches_resolution)
        self.siem4 = SIEM(embed_dim=embed_dim, patches_resolution=self.patches_resolution)
        self.quality_head = QualityRegressionHead(embed_dim=embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vgfe_features = self.vgfe(x)
        clfe_features = self.clfe(x)
        local_feature = torch.cat(clfe_features, dim=-1)
        local_feature = self.local_projection(self.local_cwsa(local_feature.transpose(1, 2)).transpose(1, 2))

        feature = self.siem1(self.cwsa1(torch.cat((vgfe_features[0], local_feature), dim=-1).transpose(1, 2)))
        feature = self.siem2(self.cwsa2(torch.cat((vgfe_features[1], feature), dim=-1).transpose(1, 2)))
        feature = self.siem3(self.cwsa3(torch.cat((vgfe_features[2], feature), dim=-1).transpose(1, 2)))
        feature = self.siem4(self.cwsa4(torch.cat((vgfe_features[3], feature), dim=-1).transpose(1, 2)))

        quality_feature = torch.mean(feature, dim=1)
        return self.quality_head(quality_feature)


IQAModel = GlintIQA
