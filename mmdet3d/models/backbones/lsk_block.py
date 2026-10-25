import torch
import torch.nn as nn


class LSKBlock(nn.Module):
    """Large Selective Kernel block (Li et al., LSKNet).

    Two depthwise paths (5x5 and dilated-7x7-on-5x5) are gated by a
    learned spatial attention map.  The composition spans a 23x23
    receptive field.  A zero-initialized scalar gate wraps the whole
    block so that the network starts as an identity (ReZero convention).
    """

    def __init__(self, dim):
        super().__init__()
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.dw7d3 = nn.Conv2d(dim, dim, 7, padding=9, dilation=3, groups=dim)

        self.pw1 = nn.Conv2d(dim, dim, 1)
        self.pw2 = nn.Conv2d(dim, dim, 1)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2 * dim, 2, 7, padding=3),
            nn.Sigmoid(),
        )
        self.pw_out = nn.Conv2d(dim, dim, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        a_shared = self.dw5(x)
        a1 = self.pw1(a_shared)
        a2 = self.pw2(self.dw7d3(a_shared))

        a_cat = torch.cat([a1, a2], dim=1)               # 2C
        a_mean = torch.mean(a_cat, dim=1, keepdim=True)   # 1
        a_max, _ = torch.max(a_cat, dim=1, keepdim=True)  # 1
        s = self.spatial_gate(torch.cat([a_mean.expand_as(a1),
                                         a_max.expand_as(a1)], dim=1))
        s1, s2 = s[:, 0:1], s[:, 1:2]

        out = self.pw_out(s1 * a1 + s2 * a2)
        return x + self.alpha * (x * out)


class ImageGuidance(nn.Module):
    """Restore the cross-modal connection severed by replacing DCNv3
    with LSK (Eq. 9 in the paper).  Two 1x1 convolutions with BN+Sigmoid
    in between, gated by a zero-initialized scalar beta."""

    def __init__(self, img_channels, out_channels):
        super().__init__()
        self.w1 = nn.Conv2d(img_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.w2 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, g):
        return x + self.beta * self.w2(torch.sigmoid(self.bn(self.w1(g))))
