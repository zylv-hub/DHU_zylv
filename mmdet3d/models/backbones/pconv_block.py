import torch
import torch.nn as nn


class PConvBlock(nn.Module):
    """Partial Convolution residual block (Chen et al., FasterNet).

    Only the first C/n channels are convolved; the rest pass through.
    Block: PConv -> BN -> ReLU -> 1x1 -> BN, with a residual connection.
    """

    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv

        self.partial_conv = nn.Conv2d(self.dim_conv, self.dim_conv, 3,
                                      padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        identity = x
        x1, x2 = x[:, :self.dim_conv], x[:, self.dim_conv:]
        x1 = self.partial_conv(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.relu(self.bn1(x))
        x = self.bn2(self.pw(x))
        return self.relu(x + identity)
