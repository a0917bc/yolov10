"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import torch
import torchvision


def conv_3x3_bn(inp, oup, stride):
    # print(inp, oup, stride)
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    # print(inp, oup)
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # print(inp, oup, stride, expand_ratio)
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


import torch.nn as nn
from timm.layers import DropPath


class TanhApprox(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x * (1 - (x * x) / 128)) / 4

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # f(x) = (1/4) * x * (1 - x^2/128)
        # df/dx = (1/4) * [ (1 - x^2/128) + x * (-2x / 128) ]
        #        = (1/4) * [1 - x^2/128 - (2x^2 / 128)]
        #        = (1/4) * [1 - 3x^2/128]
        grad_input = grad_output * (1 - 3 * x * x / 128) / 4
        return grad_input


class DynamicTanh(nn.Module):
    def __init__(self, c, alpha=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha)
        self.gamma = nn.Parameter(torch.ones(c))
        self.beta = nn.Parameter(torch.ones(c))

    def forward(self, x):
        return (
            self.gamma[:, None, None] * TanhApprox.apply(x * self.alpha)
            + self.beta[:, None, None]
        )


class ConvBNReLU(nn.Sequential):
    def __init__(
        self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, q=True
    ):
        padding = (kernel_size - 1) // 2
        if q:
            super(ConvBNReLU, self).__init__(
                torch.quantization.QuantStub(),
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes, momentum=0.1),
                nn.ReLU(inplace=False),
            )
        else:
            super(ConvBNReLU, self).__init__(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size,
                    stride,
                    padding,
                    groups=groups,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes, momentum=0.1),
                nn.ReLU(inplace=False),
            )


class Graystem(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        # print(in_planes, out_planes, kernel_size, stride, groups)
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes, momentum=0.1),
            nn.ReLU(inplace=False),
        )
        self.gray = torchvision.transforms.Grayscale()

    def forward(self, x):
        x = self.gray(x)
        x = self.conv(x)
        return x


class InvertedResidualalcor(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, drop_path=0):
        super(InvertedResidualalcor, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, q=False))

        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, q=False
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, momentum=0.1),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.skip_add = nn.quantized.FloatFunctional()
        # self.norm = DynamicTanh(c = oup)

    def forward(self, x):
        if self.use_res_connect:
            return self.skip_add.add(x, self.drop_path(self.conv(x)))
            # return self.norm(self.skip_add.add(x, self.drop_path(self.conv(x))))
        else:
            x = self.conv(x)
            return x
