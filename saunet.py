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
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, 2, 1, bias=False)

    def forward(self, x):
        skip_feature = self.doubleConv(x)
        result = self.downsample(skip_feature)
        return result, skip_feature


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.doubleConv = DoubleConv(in_channels, 2 * out_channels)
        self.upsample = nn.ConvTranspose2d(2 * out_channels, out_channels, 3, 2, 1, 1, bias=False)

    def forward(self, x, skip_feature):
        x = self.doubleConv(x)
        x = self.upsample(x)
        result = torch.cat([x, skip_feature], dim=1)
        return result


class Encoder(nn.Module):
    def __init__(self, in_channels, base_dim):
        super(Encoder, self).__init__()
        out_channels = [base_dim * 2 ** i for i in range(3)]
        self.down1 = Down(in_channels, out_channels[0])
        self.down2 = Down(out_channels[0], out_channels[1])
        self.down3 = Down(out_channels[1], out_channels[2])

    def forward(self, x):
        skip_features = []
        x, skip_feature = self.down1(x)
        skip_features.append(skip_feature)
        x, skip_feature = self.down2(x)
        skip_features.append(skip_feature)
        result, skip_feature = self.down3(x)
        skip_features.append(skip_feature)
        return result, skip_features


class SelfAttention(nn.Module):
    def __init__(self, base_dim, drop_out):
        super(SelfAttention, self).__init__()
        channel = base_dim * 2 ** 2
        self.conv1_1 = nn.Conv2d(channel, channel, 1, 1, bias=False)
        self.conv1_2 = nn.Conv2d(channel, channel, 1, 1, bias=False)
        self.conv1_3 = nn.Conv2d(channel, channel, 1, 1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        if drop_out:
            self.dropout = nn.Dropout(p=0.3)
        else:
            self.dropout = None
        self.conv2Block = nn.Sequential(
            nn.Conv2d(channel, channel, 1, 1, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        k = self.conv1_1(x)
        q = self.conv1_2(x)
        v = self.conv1_3(x)
        B, C, H, W = k.shape
        k = torch.reshape(k, (B, C, -1))
        q = torch.reshape(q, (B, C, -1))
        v = torch.reshape(v, (B, C, -1))
        k = torch.permute(k, (0, 2, 1))
        v = torch.permute(v, (0, 2, 1))
        map = self.softmax(torch.matmul(k, q))
        if self.dropout is not None:
            map = self.dropout(map)
        att_v = torch.matmul(map, v)
        att_v = torch.permute(att_v, (0, 2, 1))
        att_v = torch.reshape(att_v, (B, C, H, W))
        att_v = self.conv2Block(att_v)
        result = att_v + x
        return result


class Decoder(nn.Module):
    def __init__(self, base_dim):
        super(Decoder, self).__init__()
        out_channels = [base_dim * 2 ** i for i in range(4)]
        self.up1 = Up(out_channels[2], out_channels[2])
        self.up2 = Up(out_channels[3], out_channels[1])
        self.up3 = Up(out_channels[2], out_channels[0])

    def forward(self, x, skip_features):
        x = self.up1(x, skip_features[2])
        x = self.up2(x, skip_features[1])
        result = self.up3(x, skip_features[0])
        return result


class Output(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Output, self).__init__()
        self.doubleConv = DoubleConv(in_channels, in_channels // 2)
        self.out = nn.Conv2d(in_channels // 2, num_classes, 1, 1, bias=False)

    def forward(self, x):
        x = self.doubleConv(x)
        result = self.out(x)
        return result


class SAUNet(nn.Module):
    def __init__(self, in_channels=3, base_dim=64, num_classes=2, self_attn=True, drop_out=False):
        super(SAUNet, self).__init__()
        self.encoder = Encoder(in_channels, base_dim)
        if self_attn:
            self.bottleNeck = SelfAttention(base_dim, drop_out)
        else:
            self.bottleNeck = None
        self.decoder = Decoder(base_dim)
        self.out = Output(base_dim * 2, num_classes)

    def forward(self, x):
        x, skip_features = self.encoder(x)
        if self.bottleNeck is not None:
            x = self.bottleNeck(x)
        x = self.decoder(x, skip_features)
        result = self.out(x)
        return result
