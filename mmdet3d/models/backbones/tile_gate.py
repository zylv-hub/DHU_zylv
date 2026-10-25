import torch
import torch.nn as nn
import torch.nn.functional as F


class TileGate(nn.Module):
    """Tile-level foreground gate (Eq. 6-8 in the paper).

    Partitions the BEV grid into t x t tiles, reads the mean sigmoid
    mask over each tile, and gates the deep branch vs. a shallow one.
    This redistributes representational depth but does not save arithmetic.
    """

    def __init__(self, tile_size=12, tau_g=0.05):
        super().__init__()
        self.tile_size = tile_size
        self.tau_g = tau_g

    def forward(self, f_base, f_deep, bev_mask):
        """
        Args:
            f_base: (B, C, H, W) output of the first block (shallow).
            f_deep: (B, C, H, W) output of the full cascade (deep).
            bev_mask: (B, 1, H, W) the foreground mask logits.

        Returns:
            (B, C, H, W) tile-gated feature map.
        """
        t = self.tile_size
        B, _, H, W = bev_mask.shape

        mask_sig = torch.sigmoid(bev_mask)
        ph = (t - H % t) % t
        pw = (t - W % t) % t
        if ph > 0 or pw > 0:
            mask_sig = F.pad(mask_sig, (0, pw, 0, ph))
        Hp, Wp = mask_sig.shape[2], mask_sig.shape[3]

        tile_avg = mask_sig.reshape(B, 1, Hp // t, t, Wp // t, t).mean(dim=(3, 5))
        g = (tile_avg > self.tau_g).float()
        g = g.repeat_interleave(t, dim=2).repeat_interleave(t, dim=3)
        g = g[:, :, :H, :W]

        return g * f_deep + (1 - g) * f_base
