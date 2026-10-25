"""Role-matched BEV backbone.

Replaces the Cascade-SBDB (DCNv3) backbone of BEVDilation with:
  - PConv residual blocks (partial convolution, FasterNet)
  - LSK blocks (large selective kernel, LSKNet) with zero-init gate
  - Tile gate that redistributes representational depth
  - Image guidance that restores the severed cross-modal connection
"""
import torch
import torch.nn as nn

from ..builder import BACKBONES
from .pconv_block import PConvBlock
from .lsk_block import LSKBlock, ImageGuidance
from .tile_gate import TileGate
from .bev_backbone_ded import DEDBackbone


class OursStage(nn.Module):
    """One stage of the role-matched BEV backbone.

    Each stage: optional downsample -> num_pconv PConvBlocks -> 1 LSKBlock
    with image guidance.
    """

    def __init__(self, dim, num_pconv, stride=1, img_channels=32):
        super().__init__()
        layers = []
        if stride > 1:
            layers.append(nn.Sequential(
                nn.Conv2d(dim, dim, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            ))
        for _ in range(num_pconv):
            layers.append(PConvBlock(dim))
        self.pconv_layers = nn.Sequential(*layers)
        self.lsk = LSKBlock(dim)
        self.guide = ImageGuidance(img_channels, dim)

    def forward(self, x, img_feat):
        x = self.pconv_layers(x)
        x = self.lsk(x)
        x = self.guide(x, img_feat)
        return x


@BACKBONES.register_module()
class RoleMatchedBEVBackbone(nn.Module):
    """BEV backbone with PConv + LSK + tile gate + image guidance.

    Follows the same 4-stage encoder-decoder topology as Cascade_SBDB but
    replaces DCNv3 with lightweight operators.

    Args:
        num_layers (int): Number of cascade repetitions (kept=4 for compat).
        num_SBB (list[int]): PConv blocks per stage (e.g. [2, 1, 1]).
        down_strides (list[int]): Downsample stride per stage.
        feature_dim (int): BEV feature dimension.
        input_channels (int): Input channel count from sparse encoder.
        img_channels (int): Image BEV channel count (default 32).
        tile_size (int): Tile gate granularity (default 12).
        tau_g (float): Tile gate threshold (default 0.05).
    """

    def __init__(self,
                 num_layers=4,
                 num_SBB=[2, 1, 1],
                 down_strides=[1, 2, 2],
                 feature_dim=256,
                 input_channels=192,
                 img_channels=32,
                 tile_size=12,
                 tau_g=0.05,
                 **kwargs):
        super().__init__()
        dim = feature_dim

        self.input_proj = None
        if input_channels != dim:
            self.input_proj = nn.Sequential(
                nn.Conv2d(input_channels, dim, 3, padding=1, bias=False),
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True),
            )

        self.stages = nn.ModuleList()
        num_levels = len(down_strides)
        for idx in range(num_levels):
            stride = down_strides[idx]
            n_pconv = num_SBB[idx]
            self.stages.append(OursStage(dim, n_pconv, stride, img_channels))

        self.decoder = nn.ModuleList()
        self.decoder_norm = nn.ModuleList()
        for idx in range(num_levels - 1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(dim, dim, down_strides[idx],
                                   down_strides[idx], bias=False),
                nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))
            self.decoder_norm.append(
                nn.BatchNorm2d(dim, eps=1e-3, momentum=0.01))

        self.tile_gate = TileGate(tile_size, tau_g)
        self.num_bev_features = dim

        self.img_layer = DEDBackbone(
            [1] * num_levels, down_strides, img_channels, img_channels)

        self.num_layers = num_layers
        self.init_weights()

    def init_weights(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _run_one_pass(self, x, img_feats_list):
        """Single encoder-decoder pass."""
        feats = []
        for idx, stage in enumerate(self.stages):
            img_f = img_feats_list[idx] if idx < len(img_feats_list) else \
                img_feats_list[-1]
            x = stage(x, img_f)
            feats.append(x)

        for deconv, norm, up_x in zip(
                self.decoder, self.decoder_norm, feats[:-1][::-1]):
            x = norm(deconv(x) + up_x)
        return x

    def forward(self, x, img_feats=None, bev_mask=None):
        """
        Args:
            x: (B, C_in, H, W) BEV feature from sparse encoder.
            img_feats: (B, 32, H, W) image BEV.
            bev_mask: (B, 1, H, W) foreground mask logits (for tile gate).
        """
        if self.input_proj is not None:
            x = self.input_proj(x)

        img_feats_list = self.img_layer(img_feats)
        if not isinstance(img_feats_list, list):
            img_feats_list = [img_feats_list]

        f_base = self._run_one_pass(x, img_feats_list)

        if bev_mask is not None and self.num_layers > 1:
            f_deep = f_base
            for _ in range(self.num_layers - 1):
                f_deep = self._run_one_pass(f_deep, img_feats_list)
            out = self.tile_gate(f_base, f_deep, bev_mask)
        else:
            out = f_base

        return [out]
