# cell-unet 实现
import torch
import torch.nn as nn


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
        result = self.convBlock2(x)
        return result


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.doubleConv = DoubleConv(in_channels, out_channels)
        self.pixelConv = nn.PixelShuffle(upscale_factor=2)
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False)

    def forward(self, x):
        x1 = self.doubleConv(x)
        skip_feature = self.pixelConv(x1)
        result = self.downsample(x1)
        return result, skip_feature


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1)
        self.doubleConv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_feature):
        x = self.upsample(x)
        x = torch.cat([x, skip_feature], dim=1)
        result = self.doubleConv(x)
        return result


class Encoder(nn.Module):
    def __init__(self, in_channels, base_dim):
        super(Encoder, self).__init__()
        out_channels = [base_dim * 2 ** i for i in range(4)]
        self.down1 = Down(in_channels, out_channels[0])
        self.down2 = Down(out_channels[0], out_channels[1])
        self.down3 = Down(out_channels[1], out_channels[2])
        self.down4 = Down(out_channels[2], out_channels[3])

    def forward(self, x):
        skip_features = []
        x, skip_feature = self.down1(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down2(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down3(x)
        skip_features.append(skip_feature)
        result, skip_feature = self.down4(x)
        skip_features.append(skip_feature)
        return result, skip_features


class BottleNeck(nn.Module):
    def __init__(self, base_dim):
        super(BottleNeck, self).__init__()
        in_channels = base_dim * 2 ** 3
        out_channels = base_dim * 2 ** 4
        self.doubleConv = DoubleConv(in_channels, out_channels)
        self.pixelConv = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x = self.doubleConv(x)
        result = self.pixelConv(x)
        return result


class Decoder(nn.Module):
    def __init__(self, base_dim):
        super(Decoder, self).__init__()
        base_dim = base_dim // 4
        out_channels = [base_dim * 2 ** (4 - i) for i in range(5)]
        self.up1 = Up(out_channels[0], out_channels[1])
        self.up2 = Up(out_channels[1], out_channels[2])
        self.up3 = Up(out_channels[2], out_channels[3])
        self.up4 = Up(out_channels[3], out_channels[4])

    def forward(self, x, skip_features):
        x = self.up1(x, skip_features[3])
        x = self.up2(x, skip_features[2])
        x = self.up3(x, skip_features[1])
        result = self.up4(x, skip_features[0])
        return result


class Output(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Output, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.out = nn.Conv2d(in_channels, num_classes, 1, 1, bias=False)

    def forward(self, x):
        x = self.downsample(x)
        result = self.out(x)
        return result


class CellUNet(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, num_classes=2):
        super(CellUNet, self).__init__()
        self.encoder = Encoder(in_channels, base_dim)
        self.bottleNeck = BottleNeck(base_dim)
        self.decoder = Decoder(base_dim)
        self.out = Output(base_dim // 4, num_classes)

    def forward(self, x):
        x, skip_features = self.encoder(x)
        x = self.bottleNeck(x)
        x = self.decoder(x, skip_features)
        result = self.out(x)
        return result
