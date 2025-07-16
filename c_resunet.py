# c-ResUnet实现
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
            nn.ELU(inplace=True)
        )
        self.convBlock2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.convBlock1(x)
        result = self.convBlock2(x)
        return result


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mode='conv', residual=True):
        super(Down, self).__init__()
        self.doubleConv = DoubleConv(in_channels, out_channels)
        if residual:
            self.resBlock = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        else:
            self.resBlock = None
        if mode == 'conv':
            self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False)
        else:
            self.downsample = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip_feature = self.doubleConv(x)
        if self.resBlock is not None:
            skip_feature = skip_feature + self.resBlock(x)
        result = self.downsample(skip_feature)
        return result, skip_feature


class Encoder(nn.Module):
    def __init__(self, in_channels, base_dim, mode):
        super(Encoder, self).__init__()
        out_channels = [base_dim * 2 ** i for i in range(3)]
        self.down1 = Down(in_channels, out_channels[0], mode, False)
        self.down2 = Down(out_channels[0], out_channels[1], mode, True)
        self.down3 = Down(out_channels[1], out_channels[2], mode, True)

    def forward(self, x):
        skip_features = []
        x, skip_feature = self.down1(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down2(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down3(x)
        skip_features.append(skip_feature)
        return x, skip_features


class BottleNeck(nn.Module):
    def __init__(self, base_dim):
        super(BottleNeck, self).__init__()
        in_channels = base_dim * 2 ** 2
        out_channels = base_dim * 2 ** 3
        self.doubleConv1 = DoubleConv(in_channels, out_channels)
        self.resBlock1 = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.doubleConv2 = DoubleConv(out_channels, out_channels)
        self.resBlock2 = nn.Conv2d(out_channels, out_channels, 1, 1, bias=False)

    def forward(self, x):
        x = self.doubleConv1(x) + self.resBlock1(x)
        result = self.doubleConv2(x) + self.resBlock2(x)
        return result


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False, residual=False):
        super(Up, self).__init__()
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1, 1)
        self.residual = residual
        self.doubleConv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_feature):
        x1 = self.upsample(x)
        x = torch.cat([x1, skip_feature], dim=1)
        result = self.doubleConv(x)
        if self.residual:
            result = result + x1
        return result


class Decoder(nn.Module):
    def __init__(self, base_dim, bilinear=False):
        super(Decoder, self).__init__()
        out_channels = [base_dim * 2 ** (3 - i) for i in range(4)]
        self.up1 = Up(out_channels[0], out_channels[1], bilinear, True)
        self.up2 = Up(out_channels[1], out_channels[2], bilinear, True)
        self.up3 = Up(out_channels[2], out_channels[3], bilinear, True)

    def forward(self, x, skip_features):
        x = self.up1(x, skip_features[2])
        x = self.up2(x, skip_features[1])
        result = self.up3(x, skip_features[0])
        return result


class CResUNet(nn.Module):
    def __init__(self, in_channels=3, base_dim=16, num_classes=2, bilinear=False, mode='conv'):
        super(CResUNet, self).__init__()
        self.encoder = Encoder(in_channels, base_dim, mode)
        self.bottleNeck = BottleNeck(base_dim)
        self.decoder = Decoder(base_dim, bilinear)
        self.out = nn.Conv2d(base_dim, num_classes, 1, 1, bias=False)

    def forward(self, x):
        x, skip_features = self.encoder(x)
        x = self.bottleNeck(x)
        x = self.decoder(x, skip_features)
        result = self.out(x)
        return result
