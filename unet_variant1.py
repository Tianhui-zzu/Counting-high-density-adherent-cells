from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import SE


class MKConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MKConv, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv1_3 = nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False)

        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x3 = self.conv1_3(x)
        x = x1 + x2 + x3
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return x


class DoubleConv_variant(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv_variant, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.convBlock2 = MKConv(mid_channels, out_channels)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        return x


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


class Down_variant(nn.Module):
    def __init__(self, in_channels, out_channels, mode='conv'):
        """
            Args:
                in_channels: the dim of input feature
                out_channels: the dim of output feature
                mode: the mode of downsample, including conv(stride=2) and maxpooling
        """
        super(Down_variant, self).__init__()
        self.mode = mode
        self.doubleConv = DoubleConv_variant(in_channels, out_channels)
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


class Multi_Scale_Feature_Fusion(nn.Module):
    def __init__(self, base_dim):
        super(Multi_Scale_Feature_Fusion, self).__init__()
        channels = [base_dim * 2 ** i for i in range(4)]
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(channels[3], channels[2], 2, 2, bias=False),
            nn.ConvTranspose2d(channels[2], channels[1], 2, 2, bias=False),
            nn.ConvTranspose2d(channels[1], channels[0], 2, 2, bias=False),
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(channels[2], channels[1], 2, 2, bias=False),
            nn.ConvTranspose2d(channels[1], channels[0], 2, 2, bias=False),
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(channels[1], channels[0], 2, 2, bias=False),
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False)
        )
        self.se = SE(channels[0], channels[0])
        self.outConv = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, decoder_features):
        for i in range(len(decoder_features)):
            if i == 0:
                decoder_features[i] = self.up1(decoder_features[i])
            elif i == 1:
                decoder_features[i] = self.up2(decoder_features[i])
            elif i == 2:
                decoder_features[i] = self.up3(decoder_features[i])
            else:
                continue
        result = decoder_features[0] + decoder_features[1] + decoder_features[2] + decoder_features[3]
        result = self.se(result)
        result = self.outConv(result)
        return result


class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, mode='conv'):
        super(Encoder, self).__init__()
        channels = [base_dim * 2 ** i for i in range(4)]
        self.down1 = Down_variant(in_channels, channels[0])
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
        self.msff = Multi_Scale_Feature_Fusion(base_dim)

    def forward(self, x, skip_features):
        decoder_features = []
        x = self.up1(x, skip_features[3])
        decoder_features.append(x)
        x = self.up2(x, skip_features[2])
        decoder_features.append(x)
        x = self.up3(x, skip_features[1])
        decoder_features.append(x)
        x = self.up4(x, skip_features[0])
        decoder_features.append(x)
        x = self.msff(decoder_features)
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


class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.out = nn.Conv2d(in_channels, num_classes, 1, 1)

    def forward(self, x):
        x = self.out(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, num_classes=2, bilinear=False, mode='conv'):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, base_dim, mode)
        self.bottleNeck = BottleNeck(base_dim)
        self.decoder = Decoder(base_dim, bilinear)
        self.out = OutConv(base_dim, num_classes)

    def forward(self, x):
        x, skip_features = self.encoder(x)
        x = self.bottleNeck(x)
        x = self.decoder(x, skip_features)
        result = self.out(x)
        return result
