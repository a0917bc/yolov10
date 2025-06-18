import torch
import torch.nn as nn
from mobilenetv2 import InvertedResidual, MobileNetV2, ConvBNReLU, make_divisible


class AlcorNetV1(MobileNetV2):
    def __init__(self, num_channels=1, num_classes=16, width_mult=1.0, inverted_residual_setting=None, round_nearest=8, pretrained=False):
        """
        AlcorNet V1 main class

        Args:
            num_channels(int): (color) inputs channel, 3 for color, 1 for mono
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """

        # Call initialize from MobileNetV2 super class (nn.Module)
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 16
        last_channel = 40

        #setting up/initialize inverted residual blocks
        block_setting = [
          # t,  c,  s (t->expand ratio, c->channel, s->stride)
            [1, 8, 1],
            [10, 20, 2],
            [3.6, 20, 1],
            [5.4, 24, 2],
            [6, 24, 1],
            [6, 48, 2],
            [4, 48, 1],
            [2, 24, 1],
            [6, 24, 1],
            [6, 40, 2]
        ]

        # building first layer
        input_channel = \
            make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = \
            make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(num_channels, input_channel, stride=2)]

        # building inverted residual blocks in middle layer
        for t, c, s in block_setting:
            output_channel = c
            features.append(block(input_channel, output_channel, stride=s, expand_ratio=t))
            input_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
        #self.head = nn.Sequential(
        #    nn.Dropout(0.2),
        #    nn.Linear(self.last_channel, 3),
        #)

        self.init_weight()

    #@torch.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": True})
    def forward(self, x):
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x