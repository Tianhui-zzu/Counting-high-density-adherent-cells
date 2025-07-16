from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import ChannelAttention, SpatialAttention


# 仅加入多尺度特征融合模块

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Module):

    def __init__(self, in_channels, out_channels, mode='conv'):
        """
            Args:
                in_channels: the dim of input feature
                out_channels: the dim of output feature
                mode: the mode of downsample, including conv(stride=2) and maxpooling
        """
        super(Down, self).__init__()
        self.mode = mode
        self.doubleConv = DoubleConv(in_channels, out_channels)
        if mode == 'conv':
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False)
        else:
            self.downsample = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        skip_feature = self.doubleConv(x)
        result = self.downsample(skip_feature)
        return result, skip_feature


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, mode='conv'):
        super(Encoder, self).__init__()
        channels = [base_dim * 2 ** i for i in range(4)]
        self.down1 = Down(in_channels, channels[0])
        self.down2 = Down(channels[0], channels[1])
        self.down3 = Down(channels[1], channels[2])
        self.down4 = Down(channels[2], channels[3])

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        skip_features = []
        x, skip_feature = self.down1(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down2(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down3(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down4(x)
        skip_features.append(skip_feature)
        return x, skip_features


class Decoder(nn.Module):
    def __init__(self, base_dim=64, bilinear=False):
        super(Decoder, self).__init__()
        out_channels = [base_dim * 2 ** i for i in range(5)]
        self.up1 = Up(out_channels[4], out_channels[3], bilinear)
        self.up2 = Up(out_channels[3], out_channels[2], bilinear)
        self.up3 = Up(out_channels[2], out_channels[1], bilinear)
        self.up4 = Up(out_channels[1], out_channels[0], bilinear)

    def forward(self, x, skip_features):
        x = self.up1(x, skip_features[3])
        x = self.up2(x, skip_features[2])
        x = self.up3(x, skip_features[1])
        x = self.up4(x, skip_features[0])
        return x


class BottleNeck(nn.Module):
    def __init__(self, base_dim=64):
        super(BottleNeck, self).__init__()
        in_channels = base_dim * 2 ** 3
        out_channels = base_dim * 2 ** 4
        self.doubleConv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.doubleConv(x)
        return x


class MSFF(nn.Module):
    def __init__(self, in_channels1, in_channels2):
        super(MSFF, self).__init__()
        in_channels = in_channels1 + in_channels2
        self.sa1 = SpatialAttention()
        self.ca1 = ChannelAttention(in_channels)
        self.sa2 = SpatialAttention()
        self.ca2 = ChannelAttention(in_channels)
        self.conv1 = nn.Conv2d(in_channels, in_channels1, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels2, 3, 1, 1, bias=False)

    def forward(self, x1, x2):
        shape1 = x1.shape[-1]
        shape2 = x2.shape[-1]
        x2_1 = nn.functional.interpolate(x2, (shape1, shape1), mode='bilinear', align_corners=True)
        x1_2 = nn.functional.adaptive_avg_pool2d(x1, (shape2, shape2))
        x1_1 = torch.cat((x1, x2_1), dim=1)
        x2_2 = torch.cat((x2, x1_2), dim=1)
        x1_1 = x1_1 + self.ca1(self.sa1(x1_1))
        x2_2 = x2_2 + self.ca1(self.sa1(x2_2))
        x1 = self.conv1(x1_1)
        x2 = self.conv2(x2_2)
        return x1, x2


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Baseline1(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, num_classes=2, bilinear=False, mode='conv'):
        super(Baseline1, self).__init__()
        channels = [base_dim * 2 ** i for i in range(4)]
        self.encoder = Encoder(in_channels, base_dim, mode)
        self.msff1 = MSFF(channels[0], channels[1])
        self.msff2 = MSFF(channels[2], channels[3])
        self.bottleNeck = BottleNeck(base_dim)
        self.decoder = Decoder(base_dim, bilinear)
        self.out = OutConv(base_dim, num_classes)

    def forward(self, x):
        x, skip_features = self.encoder(x)
        new_skip_features = []
        x1, x2 = self.msff1(skip_features[0], skip_features[1])
        new_skip_features.append(x1)
        new_skip_features.append(x2)
        x1, x2 = self.msff2(skip_features[2], skip_features[3])
        new_skip_features.append(x1)
        new_skip_features.append(x2)
        x = self.bottleNeck(x)
        x = self.decoder(x, new_skip_features)
        result = self.out(x)
        return result
