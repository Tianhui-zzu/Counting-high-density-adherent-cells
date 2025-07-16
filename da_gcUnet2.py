"""
da_gcUnet
unet add dual attention and gcVit module
dual attention include channel attention and spatial attention
write by dxl
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from timm.models.layers import trunc_normal_, DropPath

import train_utils.configs as configs
from blocks import CoordAttention as CA
from blocks import DA_Module, Block, SimAM_module
from pytorch_wavelets import DWTForward


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


def _to_channel_last(x):
    """
    Args:
        x:(B, C, H, W)
    Returns:
        x:(B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x:(B, H, W, C)
    Returns:
        x:(B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size, h_w, w_w):
    """
    Args:
        x:(B, H, W, C)
        window_size: window's size
        h_w: the number of window height
        w_w: the number of window width
    Returns:
        local window features(num_window*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, h_w, window_size, w_w, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def windows_reverse(windows, window_size, H, W, h_w, w_w, B):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x:(B, H, W, C)
    """
    windows = windows.view(B, h_w, w_w, window_size, window_size, -1)
    x = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(mid_channels)
        self.gelu1 = nn.GELU()
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(out_channels)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchNorm1(x)
        x = self.gelu1(x)
        x = self.conv2(x)
        x = self.batchNorm2(x)
        x = self.gelu2(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.deConv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1,
                                          output_padding=1, bias=False)
        # self.deConv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        self.batchNorm1 = nn.BatchNorm2d(out_channels)
        self.gelu1 = nn.GELU()

    def forward(self, x):
        x = self.deConv1(x)
        x = self.batchNorm1(x)
        x = self.gelu1(x)
        return x


class Final_UpSample(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Final_UpSample, self).__init__()
        self.doubleConv1 = DoubleConv(in_channels, mid_channels)
        self.up1 = Up(mid_channels, mid_channels)
        self.doubleConv2 = DoubleConv(mid_channels, out_channels)
        self.up2 = Up(out_channels, out_channels)

    def forward(self, x_encoder, x_decoder):
        x = self.doubleConv1(x_decoder)
        x = self.up1(x)
        x = self.doubleConv2(x)
        x = self.up2(x)
        result = torch.cat((x, x_encoder), dim=1)
        return result


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.upSample = Up(in_channels, in_channels // 2)
        self.doubleConv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_feature):
        x = self.upSample(x)
        x = torch.cat((x, skip_feature), dim=1)
        result = self.doubleConv(x)
        return result


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inp, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ReduceSize(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """
        super(ReduceSize, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            # CA(dim, dim),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim_out)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        x = x + self.conv(x)
        x = self.reduction(x)
        x = _to_channel_last(x)
        x = self.norm2(x)
        return x


class DWC_Module(nn.Module):
    def __init__(self, in_channels):
        super(DWC_Module, self).__init__()
        self.dwBlock = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.GELU()
        )
        self.seBlock = nn.Sequential(
            SE(in_channels, in_channels),
            nn.Conv2d(in_channels, in_channels, 1, 1, bias=False)
        )

    def forward(self, x):
        shortcut1 = x
        x = self.dwBlock(x)
        x = self.seBlock(x)
        x = x + shortcut1
        return x


class ReduceSize_variant(nn.Module):
    def __init__(self, dim, dim_out=None, norm_layer=nn.LayerNorm, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """
        super(ReduceSize_variant, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            # CA(dim, dim),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        )
        if keep_dim:
            dim_out = dim
        else:
            if dim_out is None:
                dim_out = 2 * dim
            else:
                dim_out = dim_out
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 1, 1),
            nn.BatchNorm2d(dim_out),
            nn.GELU()
        )
        self.reduction = nn.Conv2d(dim_out, dim_out, 3, 2, 1, bias=False)
        self.norm1 = norm_layer(dim)
        # self.norm2 = norm_layer(dim_out)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        x = x + self.conv1(x)
        skip_feature = self.conv2(x)
        x = self.reduction(skip_feature)
        x = _to_channel_last(x)
        # x = self.norm2(x)
        return x, skip_feature


class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super(PatchEmbed, self).__init__()
        self.doubleConv = DoubleConv(in_channels=in_chans, out_channels=64)
        self.proj = nn.Conv2d(64, dim, 3, 2, 1)
        self.conv_down = ReduceSize(dim=dim, keep_dim=True)

    def forward(self, x):
        """
        Args:
            x(B, C, H, W)
        Return:
            x(B, H/4, W/4, dim)
        """
        x = self.doubleConv(x)
        skip_feature = x
        x = self.proj(x)
        x = _to_channel_last(x)
        x = self.conv_down(x)
        return x, skip_feature


class PatchEmbed_variant(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super(PatchEmbed_variant, self).__init__()
        mid_channels = embed_dim // 2
        self.doubleConv1 = DoubleConv(in_channels, mid_channels)
        self.downBlock1 = Down_wt(mid_channels, mid_channels)
        self.doubleConv2 = DoubleConv(mid_channels, embed_dim)
        self.downBlock2 = Down_wt(embed_dim, embed_dim)
        # self.downBlock1 = ReduceSize_variant(dim=in_channels, dim_out=mid_channels)
        # self.downBlock2 = ReduceSize_variant(dim=mid_channels, dim_out=embed_dim)

    def forward(self, x):
        skip_features = []
        skip_feature = self.doubleConv1(x)
        skip_features.append(skip_feature)
        x = self.downBlock1(skip_feature)
        skip_feature = self.doubleConv2(x)
        skip_features.append(skip_feature)
        x = self.downBlock2(skip_feature)
        x = _to_channel_last(x)
        return x, skip_features


class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]  # (B, C, H/2, W/2)
        y_LH = yH[0][:, :, 1, ::]  # (B, C, H/2, W/2)
        y_HH = yH[0][:, :, 2, ::]  # (B, C, H/2, W/2)
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)  # (B, 4*C, H/2, W/2)
        x = self.conv_bn_relu(x)  # (B, out_C, H/2, W/2)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm_layer=nn.LayerNorm):
        super(DownSample, self).__init__()
        if out_channels is None:
            out_channels = in_channels * 2
        self.doubleConv = DoubleConv(in_channels, out_channels)
        self.downBlock = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.norm1 = norm_layer(in_channels)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        skip_feature = self.doubleConv(x)
        x = self.downBlock(skip_feature)
        x = _to_channel_last(x)
        return x, skip_feature


class DownSample_variant(nn.Module):
    def __init__(self, in_channels, out_channels=None, norm_layer=nn.LayerNorm):
        super(DownSample_variant, self).__init__()
        if out_channels is None:
            out_channels = in_channels * 2
        self.doubleConv = DoubleConv(in_channels, out_channels)
        self.downBlock = Down_wt(out_channels, out_channels)
        self.norm1 = norm_layer(in_channels)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = _to_channel_first(x)
        skip_feature = self.doubleConv(x)
        x = self.downBlock(skip_feature)
        x = _to_channel_last(x)
        return x, skip_feature


class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super(PatchExpand, self).__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        Args:
            x(B, H, W, C)
        Returns:
            x(B, 2H, 2W, C/4 or C/2)
        """
        B, H, W, _ = x.shape
        x = x.view(B, H * W, -1)
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, -1)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, 2 * H, 2 * W, -1)
        return x


class FeatExtract(nn.Module):
    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """
        super(FeatExtract, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        )
        if not keep_dim:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """
        super(WindowAttention, self).__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (2, h*w, h*w) -> (h*w, h*w, 2)
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = torch.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)  # (B_, num_heads, N, head_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttentionGlobal(nn.Module):
    def __init__(self, dim, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """
        Args:
            dim: feature size dimension.
            num_heads: number of attention head.
            window_size: window size.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            attn_drop: attention dropout rate.
            proj_drop: output dropout rate.
        """
        super(WindowAttentionGlobal, self).__init__()
        window_size = (window_size, window_size)
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)
        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, q_global):
        B_, N, C = x.shape
        B = q_global.shape[0]
        head_dim = torch.div(C, self.num_heads, rounding_mode='floor')
        B_dim = torch.div(B_, B, rounding_mode='floor')
        kv = self.qkv(x).reshape(B_, N, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q_global = q_global.repeat(1, B_dim, 1, 1, 1)
        q = q_global.reshape(B_, self.num_heads, N, head_dim)
        q = q * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GCViTBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm, layer_scale=None):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            num_heads: number of attention head.
            window_size: window size.
            mlp_ratio: MLP ratio.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            act_layer: activation function.
            attention: attention block type.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        super(GCViTBlock, self).__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)
        self.attn = attention(dim, num_heads=num_heads, window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0
        inp_w = torch.div(input_resolution, window_size, rounding_mode='floor')
        self.num_windows = int(inp_w * inp_w)

    def forward(self, x, q_global):
        B, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        h_w = torch.div(H, self.window_size, rounding_mode='floor')
        w_w = torch.div(W, self.window_size, rounding_mode='floor')
        x_windows = window_partition(x, self.window_size, h_w, w_w)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global)
        x = windows_reverse(attn_windows, self.window_size, H, W, h_w, w_w, B)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GlobalQueryGen(nn.Module):
    def __init__(self, dim, input_resolution, image_resolution, window_size, num_heads):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.

        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2.

        """
        super(GlobalQueryGen, self).__init__()
        if input_resolution == image_resolution // 4:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False)
            )
        elif input_resolution == image_resolution // 8:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False)
            )
        elif input_resolution == image_resolution // 16:
            if window_size == input_resolution:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=True)
                )
            else:
                self.to_q_global = nn.Sequential(
                    FeatExtract(dim, keep_dim=False)
                )
        elif input_resolution == image_resolution // 32:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )
        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')

    def forward(self, x):
        """
        Args:
            x(B, C, H, W)
        Returns:
            x(B, 1, num_heads, N(window_size * window_size), dim_head)
        """
        x = _to_channel_last(self.to_q_global(x))
        B = x.shape[0]
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x


class GCViTLayer(nn.Module):
    def __init__(self, dim, depth, input_resolution, image_resolution, num_heads, window_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, layer_scale=None):
        """
                Args:
                    dim: feature size dimension.
                    depth: number of layers in each stage.
                    input_resolution: input image resolution.
                    window_size: window size in each stage.
                    downsample: bool argument for down-sampling.
                    mlp_ratio: MLP ratio.
                    num_heads: number of heads in each stage.
                    qkv_bias: bool argument for query, key, value learnable bias.
                    qk_scale: bool argument to scaling query, key.
                    drop: dropout rate.
                    attn_drop: attention dropout rate.
                    drop_path: drop path rate.
                    norm_layer: normalization layer.
                    layer_scale: layer scaling coefficient.
                """
        super(GCViTLayer, self).__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock(dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       attention=WindowAttention if i % 2 == 0 else WindowAttentionGlobal,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                       norm_layer=norm_layer,
                       layer_scale=layer_scale
                       )
            for i in range(depth)
        ])

        # self.downsample = None if not downsample else ReduceSize_variant(dim=dim, norm_layer=norm_layer)
        self.downsample = None if not downsample else DownSample_variant(in_channels=dim, norm_layer=norm_layer)
        self.q_global_gen = GlobalQueryGen(dim, input_resolution, image_resolution, window_size, num_heads)

    def forward(self, x):
        """
        Args:
            x(B, H, W, C)
        Returns:
            x(B, H/2, W/2, 2C)
        """
        q_global = self.q_global_gen(_to_channel_first(x))
        for blk in self.blocks:
            x = blk(x, q_global)
        if self.downsample is None:
            return x, None
        x, skip_feature = self.downsample(x)
        return x, skip_feature


class GCViTLayer_bottleNeck(nn.Module):
    def __init__(self, dim, depth, input_resolution, image_resolution, num_heads, window_size, downsample=True,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, layer_scale=None):
        """
                Args:
                    dim: feature size dimension.
                    depth: number of layers in each stage.
                    input_resolution: input image resolution.
                    window_size: window size in each stage.
                    downsample: bool argument for down-sampling.
                    mlp_ratio: MLP ratio.
                    num_heads: number of heads in each stage.
                    qkv_bias: bool argument for query, key, value learnable bias.
                    qk_scale: bool argument to scaling query, key.
                    drop: dropout rate.
                    attn_drop: attention dropout rate.
                    drop_path: drop path rate.
                    norm_layer: normalization layer.
                    layer_scale: layer scaling coefficient.
                """
        super(GCViTLayer_bottleNeck, self).__init__()
        self.blocks1 = nn.ModuleList([
            GCViTBlock(dim=dim[0], input_resolution=input_resolution[0], num_heads=num_heads[0],
                       window_size=window_size[0],
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       attention=WindowAttention if i % 2 == 0 else WindowAttentionGlobal,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[0][i] if isinstance(drop_path[0], list) else drop_path[0],
                       norm_layer=norm_layer,
                       layer_scale=layer_scale
                       )
            for i in range(depth[0])
        ])
        self.blocks2 = nn.ModuleList([
            GCViTBlock(dim=dim[1], input_resolution=input_resolution[1], num_heads=num_heads[1],
                       window_size=window_size[1],
                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                       attention=WindowAttention if i % 2 == 0 else WindowAttentionGlobal,
                       drop=drop, attn_drop=attn_drop,
                       drop_path=drop_path[1][i] if isinstance(drop_path[1], list) else drop_path[1],
                       norm_layer=norm_layer,
                       layer_scale=layer_scale
                       )
            for i in range(depth[1])
        ])

        # self.downsample = None if not downsample else ReduceSize_variant(dim=dim, norm_layer=norm_layer)
        self.downsample = None if not downsample else DownSample_variant(in_channels=dim[0], norm_layer=norm_layer)
        self.q_global_gen_1 = GlobalQueryGen(dim[0], input_resolution[0], image_resolution, window_size[0],
                                             num_heads[0])
        self.q_global_gen_2 = GlobalQueryGen(dim[1], input_resolution[1], image_resolution, window_size[1],
                                             num_heads[1])
        self.ca1 = CA(dim[0], dim[0])
        self.ca2 = CA(dim[1], dim[1])
        self.upsample = Up(dim[1], dim[0])

    def forward(self, x):
        """
        Args:
            x(B, H, W, C)
        Returns:
            x(B, H/2, W/2, 2C)
        """
        x_1 = x
        x_1 = _to_channel_last(self.ca1(_to_channel_first(x_1)))
        q_global_1 = self.q_global_gen_1(_to_channel_first(x_1))
        x_2 = self.downsample(x)[0]
        x_2 = _to_channel_last(self.ca2(_to_channel_first(x_2)))
        q_global_2 = self.q_global_gen_2(_to_channel_first(x_2))
        # for blk in self.blocks:
        #     x = blk(x, q_global)
        for blk in self.blocks1:
            x_1 = blk(x_1, q_global_1)
        for blk in self.blocks2:
            x_2 = blk(x_2, q_global_2)
        x_2 = self.upsample(_to_channel_first(x_2))
        x = x_1 + _to_channel_last(x_2)
        return x


class GCViT(nn.Module):
    def __init__(self, dim, depths, window_size, mlp_ratio, num_heads, resolution=224, drop_path_rate=0.2, in_chans=3,
                 num_classes=2, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 layer_scale=None):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            resolution: input image resolution.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        super(GCViT, self).__init__()
        num_features = int(dim * 2 ** (len(depths) - 2))
        self.num_classes = num_classes
        # self.patch_embed = PatchEmbed(in_chans=in_chans, dim=dim)
        self.patch_embed = PatchEmbed_variant(in_channels=in_chans, embed_dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths) - 2):
            level = GCViTLayer(dim=int(dim * 2 ** i), depth=depths[i], num_heads=num_heads[i],
                               window_size=window_size[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                               norm_layer=norm_layer, downsample=True,
                               layer_scale=layer_scale, input_resolution=int(2 ** (-2 - i) * resolution),
                               image_resolution=resolution
                               )
            self.levels.append(level)
        length = len(depths)
        self.levels_bottleNeck = GCViTLayer_bottleNeck(dim=[dim * 2 ** 2, dim * 2 ** 3],
                                                       depth=[depths[length - 2], depths[length - 1]],
                                                       num_heads=[num_heads[length - 2], num_heads[length - 1]],
                                                       window_size=[window_size[length - 2], window_size[length - 1]],
                                                       mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                       drop_path=[dpr[sum(depths[:2]):sum(depths[:3])],
                                                                  dpr[sum(depths[:3]):sum(depths[:4])]],
                                                       norm_layer=norm_layer, downsample=True,
                                                       layer_scale=layer_scale,
                                                       input_resolution=[int(2 ** (-4) * resolution),
                                                                         int(2 ** (-5) * resolution)],
                                                       image_resolution=resolution
                                                       )
        self.norm = norm_layer(num_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x, skip_features = self.patch_embed(x)
        x = self.pos_drop(x)
        for level in self.levels:
            x, skip_feature = level(x)
            if skip_feature is not None:
                skip_features.append(skip_feature)
        x = self.levels_bottleNeck(x)
        x = self.norm(x)
        x = _to_channel_first(x)
        return x, skip_features

    def forward(self, x):
        x, skip_features = self.forward_features(x)
        return x, skip_features


class BottleNeckPosition(nn.Module):
    def __init__(self, config):
        super(BottleNeckPosition, self).__init__()
        # upSample
        self.config = config
        self.daBlock = DA_Module(config.hidden_size, config.hidden_size)
        grid_size = 14
        n_patches = (grid_size * grid_size)
        self.positionEmbedding = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        decoder_feature = self.daBlock(x)
        b, hidden, _, _ = decoder_feature.size()
        decoder_feature = decoder_feature.view(b, hidden, -1)
        decoder_feature = decoder_feature.permute(0, 2, 1)
        result = decoder_feature + self.positionEmbedding
        result = self.dropout(result)
        return result


class TransEncoder(nn.Module):
    def __init__(self, config, vis):
        super(TransEncoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for i in range(config.transformer['num_layers']):
            layer = Block(config, self.vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        weights = []
        for layer_block in self.layer:
            weight, x = layer_block(x)
            if self.vis:
                weights.append(weight)
        result = self.encoder_norm(x)
        return weights, result


class BottleNeck(nn.Module):
    def __init__(self, config, vis):
        super(BottleNeck, self).__init__()
        dim = config.dim
        orig_dim = dim * 2 ** 2
        self.convBlock = nn.Sequential(
            nn.Conv2d(orig_dim, config.hidden_size, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(config.hidden_size),
            nn.GELU()
        )
        self.embedding = BottleNeckPosition(config)
        self.transEncoder = TransEncoder(config, vis)

    def forward(self, x):
        x = self.convBlock(x)
        decoder_feature = self.embedding(x)
        weights, decoder_feature = self.transEncoder(decoder_feature)
        b, n_patch, hidden_size = decoder_feature.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        decoder_feature = decoder_feature.permute(0, 2, 1)
        decoder_feature = decoder_feature.contiguous().view(b, hidden_size, h, w)
        return weights, decoder_feature


class BottleNeck_variant(nn.Module):
    def __init__(self, embed_dim):
        super(BottleNeck_variant, self).__init__()
        in_channels = embed_dim * 2 ** 2
        out_channels = in_channels * 2
        self.doubleConv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        result = self.doubleConv(x)
        return result


class Decoder(nn.Module):
    def __init__(self, embed_dim):
        super(Decoder, self).__init__()
        dim = embed_dim // 2
        out_channels = [dim * 2 ** i for i in range(5)]
        self.ca1 = CA(out_channels[3], out_channels[3])
        self.ca2 = CA(out_channels[2], out_channels[2])
        self.ca3 = CA(out_channels[1], out_channels[1])
        self.ca4 = CA(out_channels[0], out_channels[0])
        self.up1 = UpSample(out_channels[4], out_channels[3])
        self.up2 = UpSample(out_channels[3], out_channels[2])
        self.up3 = UpSample(out_channels[2], out_channels[1])
        self.up4 = UpSample(out_channels[1], out_channels[0])

    def forward(self, x, skip_features):
        for i in range(len(skip_features)):
            if i == 0:
                skip_features[i] = self.ca4(skip_features[i])
                # skip_features[i] = skip_features[i]
            elif i == 1:
                skip_features[i] = self.ca3(skip_features[i])
            elif i == 2:
                skip_features[i] = self.ca2(skip_features[i])
            else:
                skip_features[i] = self.ca1(skip_features[i])

        x = self.up1(x, skip_features[3])
        x = self.up2(x, skip_features[2])
        x = self.up3(x, skip_features[1])
        result = self.up4(x, skip_features[0])
        return result


class PredictHead(nn.Module):
    def __init__(self, feature_dim, num_classes):
        """
        Args:
            feature_dim: input tensor dim
            num_classes: the number of classes
        """
        super(PredictHead, self).__init__()
        dim = feature_dim // 2
        self.output = nn.Conv2d(dim, num_classes, 3, 1, 1)

    def forward(self, x):
        result = self.output(x)
        return result


class DAGCUnet(nn.Module):
    def __init__(self, config, num_classes):
        super(DAGCUnet, self).__init__()
        self.gcvit = GCViT(dim=config.dim, depths=config.depths, window_size=config.window_size,
                           mlp_ratio=config.mlp_ratio, num_heads=config.num_heads, resolution=224,
                           drop_path_rate=config.drop_path_rate, layer_scale=config.layer_scale)
        self.classifier = "seg"
        self.bottleneck = BottleNeck_variant(config.dim)
        self.decoder = Decoder(config.dim)
        self.predict_head = PredictHead(config.dim, num_classes)

    def forward(self, x):
        x, skip_features = self.gcvit(x)
        feature = self.bottleneck(x)
        feature = self.decoder(feature, skip_features)
        result = self.predict_head(feature)
        return result

    def load_from(self, config, weights):
        pretrained_path = config.pretrain_ckpt
        vit_pretrained_path = config.vit_pretrain_ckpt
        if pretrained_path is not None:
            print('pretrained_path:{}'.format(pretrained_path))
            device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)['state_dict']
            print("---start load pretrained model of GCUnet encoder---")
            model_dict = self.gcvit.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            model_dict_1 = copy.deepcopy(model_dict)
            for k, v in full_dict.items():
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                  model_dict[k].shape))
                    else:
                        model_dict_1.update({k: v})
                else:
                    if 'levels.2' in k or 'levels.3' in k:
                        if 'levels.2' in k:
                            if 'downsample' in k:
                                new_k = 'levels_bottleNeck.' + '.'.join(k.split('.')[2:])
                            elif 'q_global_gen' in k:
                                new_k = 'levels_bottleNeck.q_global_gen_1.' + '.'.join(k.split('.')[3:])
                            else:
                                new_k = 'levels_bottleNeck.blocks1.' + '.'.join(k.split('.')[3:])
                            if new_k in model_dict:
                                if full_dict[k].shape != model_dict[new_k].shape:
                                    print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                              model_dict[new_k].shape))
                                else:
                                    model_dict_1.update({new_k: v})

                            else:
                                print('{} and {} is no exists!'.format(k, new_k))


                        else:
                            if 'downsample' in k:
                                new_k = 'levels_bottleNeck.' + '.'.join(k.split('.')[2:])
                            elif 'q_global_gen' in k:
                                new_k = 'levels_bottleNeck.q_global_gen_2.' + '.'.join(k.split('.')[3:])
                            else:
                                new_k = 'levels_bottleNeck.blocks2.' + '.'.join(k.split('.')[3:])
                            if new_k in model_dict:
                                if full_dict[k].shape != model_dict[new_k].shape:
                                    print("delete:{};shape pretrain:{};shape model:{}".format(k, full_dict[k].shape,
                                                                                              model_dict[
                                                                                                  new_k].shape))
                                else:
                                    model_dict_1.update({new_k: v})
                            else:

                                print('{} and {} is not exists!'.format(k, new_k))

                    else:
                        print("{} is not exists".format(k))

            msg = self.gcvit.load_state_dict(model_dict_1, strict=False)
            print(msg)
        else:
            print("gcViT pretrain is none")

        # with torch.no_grad():
        #     res_weight = weights
        #     self.bottleneck.transEncoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
        #     # self.transformer.transEncoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
        #     self.bottleneck.transEncoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
        #     # self.transformer.transEncoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))
        #
        #     posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
        #
        #     # posemb_new = self.transformer.embeddings.position_embeddings
        #     posemb_new = self.bottleneck.embedding.positionEmbedding
        #     if posemb.size() == posemb_new.size():
        #         self.bottleneck.embedding.positionEmbedding.copy_(posemb)
        #         # self.transformer.embedding.positionEmbedding.copy_(posemb)
        #     elif posemb.size()[1] - 1 == posemb_new.size()[1]:
        #         posemb = posemb[:, 1:]
        #         self.bottleneck.embedding.positionEmbedding.copy_(posemb)
        #     else:
        #         ntok_new = posemb_new.size(1)
        #         if self.classifier == "seg":
        #             _, posemb_grid = posemb[:, :1], posemb[0, 1:]
        #         gs_old = int(np.sqrt(len(posemb_grid)))
        #         gs_new = int(np.sqrt(ntok_new))
        #         print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
        #         posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
        #         zoom = (gs_new / gs_old, gs_new / gs_old, 1)
        #         posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
        #         posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
        #         posemb = posemb_grid
        #         self.bottleneck.embedding.positionEmbedding.copy_(np2th(posemb))
        #     # Encoder whole
        #     for bname, block in self.bottleneck.transEncoder.named_children():
        #         for uname, unit in block.named_children():
        #             unit.load_from(weights, n_block=uname)


CONFIGS = {
    'GC_ViT_small': configs.get_gcvit_small_config(),
    'GC_ViT_small2': configs.get_gcvit_small2_config()
}
