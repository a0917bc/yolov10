import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
import torchvision


class NCHWLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return (super().forward(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)


class DWConvbnrelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DWDownsampleLayer(nn.Module):
    def __init__(self, in_channels, out_channels, ln=True):
        super().__init__()
        kernel_size = 7
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                groups=out_channels,
            ),
        )
        if ln:
            self.norm = NCHWLayerNorm(out_channels, eps=1e-06)
        else:
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class DWStemLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, ln=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                groups=out_channels,
            ),
        )
        if ln:
            self.norm1 = NCHWLayerNorm(out_channels, eps=1e-06)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                groups=out_channels,
            ),
        )
        if ln:
            self.norm2 = NCHWLayerNorm(out_channels, eps=1e-06)
        else:
            self.norm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DWgrayStemLayer(nn.Module):
    def __init__(self, in_channels=3, out_channels=96, ln=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                groups=out_channels,
            ),
        )
        if ln:
            self.norm1 = NCHWLayerNorm(out_channels, eps=1e-06)
        else:
            self.norm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=3 // 2,
                groups=out_channels,
            ),
        )
        if ln:
            self.norm2 = NCHWLayerNorm(out_channels, eps=1e-06)
        else:
            self.norm2 = nn.BatchNorm2d(out_channels)
        self.gray = torchvision.transforms.Grayscale()

    def forward(self, x):
        x = self.gray(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class Linear(nn.Module):
    def __init__(self, dim, num_classes=1000, bias=True, **kwargs):
        super().__init__()
        print('%15s' % 'Linear: ', dim, num_classes, bias)
        self.fc = nn.Linear(dim, num_classes, bias=bias)

    def forward(self, x):
        x = self.fc(x.permute(0, 2, 3, 1))
        return x.mean([1, 2])


class GatedCNNBlock(nn.Module):
    r"""Our implementation of Gated CNN Block: https://arxiv.org/pdf/1612.08083
    Args:
        conv_ratio: control the number of channels to conduct depthwise convolution.
            Conduct convolution on partial channels can improve paraitcal efficiency.
            The idea of partial channels is from ShuffleNet V2 (https://arxiv.org/abs/1807.11164) and
            also used by InceptionNeXt (https://arxiv.org/abs/2303.16900) and FasterNet (https://arxiv.org/abs/2303.03667)
    """

    def __init__(
        self,
        dim,
        expansion_ratio=8 / 3,
        kernel_size=7,
        conv_ratio=1.0,
        drop_path=0.0,
        **kwargs,
    ):
        super().__init__()
        hidden = int(expansion_ratio * dim)
        self.bn = nn.BatchNorm2d(hidden * 2)
        self.conv = nn.Conv2d(
            dim, hidden * 2, kernel_size=1, padding=0
        )  # nn.Linear(dim, hidden * 2)
        conv_channels = int(conv_ratio * dim)
        self.split_indices = (hidden, hidden - conv_channels, conv_channels)
        self.csp_conv = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=1, stride=1),
            nn.Conv2d(
                conv_channels,
                conv_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=conv_channels,
            ),
        )
        self.fc2 = nn.Conv2d(
            hidden, dim, kernel_size=1, padding=0
        )  # nn.Linear(hidden, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn(x)
        g, i, c = torch.split(x, self.split_indices, dim=1)
        c = self.csp_conv(c)
        x = self.fc2(F.relu(g, inplace=False) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut

    def forward_fuse(self, x):
        shortcut = x
        x = self.conv(x)
        g, i, c = torch.split(x, self.split_indices, dim=1)
        c = self.csp_conv(c)
        x = self.fc2(F.relu(g, inplace=False) * torch.cat((i, c), dim=1))
        x = self.drop_path(x)
        return x + shortcut


class ConvFormer(nn.Module):
    def __init__(self, in_ch, out_ch, strides, ratio):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.convbn = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False), nn.ReLU()
        )
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.convbnrelu = nn.Sequential(
            nn.Conv2d(in_ch, int(in_ch * ratio), 1, strides, bias=False),
            nn.ReLU(),  # ConvReLU2d
            nn.Conv2d(int(in_ch * ratio), in_ch, 1, strides, bias=False),
        )
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.skip_add.add(self.convbn(self.bn1(x)), x)
        x = self.skip_add.add(self.convbnrelu(self.bn2(x)), x)
        return x
