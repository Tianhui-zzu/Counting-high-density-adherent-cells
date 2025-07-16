import torch.nn as nn
import torch
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding, \
    LayerNorm
import math
from os.path import join as pjoin

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class PA_Module(nn.Module):
    def __init__(self, in_channels, downSample=False):
        super(PA_Module, self).__init__()
        self.downSample = downSample
        if in_channels >= 8:
            mid_channels = in_channels // 8
        else:
            mid_channels = 1
        if self.downSample:
            self.queryConv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, 1),
                nn.MaxPool2d(2, 2)
            )
            self.keyConv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=2, padding=0)
            self.valueConv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=2, padding=0)
            self.recoverConv = nn.Identity(),
        else:
            self.queryConv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
            self.keyConv = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
            self.valueConv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
            self.recoverConv = None

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        # b, c, h, w = x.size()
        query = self.queryConv(x)
        key = self.keyConv(x)
        b, _, h, w = query.size()
        b, _, h1, w1 = key.size()
        value = self.valueConv(x)
        query = query.view(b, -1, h * w).permute(0, 2, 1)
        key = key.view(b, -1, h1 * w1)
        value = value.view(b, -1, h1 * w1)
        att_map = torch.bmm(query, key)
        att_map = self.softmax(att_map)
        result = torch.bmm(value, att_map.permute(0, 2, 1))
        result = result.view(b, -1, h, w)
        if self.recoverConv is not None:
            result = nn.functional.interpolate(result, (h * 2, h * 2), mode='bilinear', align_corners=True)
        result = self.gamma * result + x
        return result


class CA_Module(nn.Module):
    def __init__(self):
        super(CA_Module, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        query = x.view(b, c, -1)
        key = x.view(b, c, -1).permute(0, 2, 1)
        value = x.view(b, c, -1)
        att_map = torch.bmm(query, key)
        new_att_map = torch.max(att_map, dim=-1, keepdim=True)[0].expand_as(att_map) - att_map
        new_att_map = self.softmax(new_att_map)
        result = torch.bmm(new_att_map, value)
        result = result.view(b, -1, h, w)
        result = self.gamma * result + x
        return result


class CoordAttention(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class DA_Module_variant(nn.Module):
    def __init__(self, in_channels, out_channels, downSample=False):
        super(DA_Module_variant, self).__init__()
        mid_channels = in_channels // 16
        self.paConv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False)
        )
        self.caConv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False)
        )
        self.pam = PA_Module(mid_channels, downSample)
        self.cam = CoordAttention(in_channels, in_channels)
        self.paConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.caConv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.sumConv = nn.Sequential(
            nn.Dropout2d(0.05, False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        pa = self.paConv1(x)
        ca = self.caConv1(x)
        pa = self.pam(pa)
        ca = self.cam(ca)
        pa = self.paConv2(pa)
        ca = self.caConv2(ca)
        da_sum = pa + ca
        result = self.sumConv(da_sum)
        return result


class DA_Module(nn.Module):
    def __init__(self, in_channels, out_channels, downSample=False):
        super(DA_Module, self).__init__()
        mid_channels = in_channels // 16
        self.paConv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False))
        self.caConv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=False))
        self.pam = PA_Module(mid_channels, downSample)
        self.cam = CA_Module()
        self.paConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
        self.caConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True))
        self.sumConv = nn.Sequential(nn.Dropout2d(0.05, False),
                                     nn.Conv2d(mid_channels, out_channels, kernel_size=1),
                                     nn.ReLU(inplace=True)
                                     )

    def forward(self, x):
        pa = self.paConv1(x)
        ca = self.caConv1(x)
        pa = self.pam(pa)
        ca = self.cam(ca)
        pa = self.paConv2(pa)
        ca = self.caConv2(ca)
        da_sum = pa + ca
        result = self.sumConv(da_sum)
        return result


"""
    input:
        embedding(b,n_patches,hidden_size)
    return:
        weights: attention weights(b,num_heads,n_patches,n_patches)
        result: attention result

"""


class SimAM_module(torch.nn.Module):
    def __init__(self, channels=None, e_lamda=1e-4):
        super(SimAM_module, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lamda = e_lamda

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lamda)) + 0.5

        return x * self.activation(y)


class Attention(nn.Module):

    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_heads = config.transformer["num_heads"]
        self.head_size = int(config.hidden_size / self.num_heads)
        self.total_head_size = self.num_heads * self.head_size
        self.queryLinear = Linear(config.hidden_size, self.total_head_size)
        self.keyLinear = Linear(config.hidden_size, self.total_head_size)
        self.valueLinear = Linear(config.hidden_size, self.total_head_size)
        self.outLinear = Linear(self.total_head_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.softmax = Softmax(dim=-1)

    def transpose_num_heads(self, x):
        b, n_patches, _ = x.size()
        x = x.view(b, n_patches, self.num_heads, self.head_size)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, x):
        query = self.queryLinear(x)
        key = self.keyLinear(x)
        value = self.valueLinear(x)
        query = self.transpose_num_heads(query)
        key = self.transpose_num_heads(key)
        value = self.transpose_num_heads(value)
        att_map = torch.matmul(query, key.transpose(-1, -2))
        att_map = att_map / math.sqrt(self.head_size)
        att_probs = self.softmax(att_map)
        weights = att_probs if self.vis else None
        att_probs = self.attn_dropout(att_probs)
        result = torch.matmul(att_probs, value)
        result = result.permute(0, 2, 1, 3)
        b, n_patches, _, _ = result.size()
        result = result.contiguous().view(b, n_patches, self.total_head_size)
        result = self.outLinear(result)
        result = self.proj_dropout(result)
        return weights, result


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.att_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.mlp_norm = LayerNorm(self.hidden_size, eps=1e-6)
        self.attention = Attention(config, vis)
        self.mlp = Mlp(config)

    def forward(self, x):
        tmp = self.att_norm(x)
        weights, tmp = self.attention(tmp)
        tmp = tmp + x
        x = tmp
        tmp = self.mlp_norm(tmp)
        tmp = self.mlp(tmp)
        result = tmp + x
        return weights, result

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            # self.attn.query.weight.copy_(query_weight)
            self.attention.queryLinear.weight.copy_(query_weight)
            # self.attn.key.weight.copy_(key_weight)
            self.attention.keyLinear.weight.copy_(key_weight)
            # self.attn.value.weight.copy_(value_weight)
            self.attention.valueLinear.weight.copy_(value_weight)
            # self.attn.out.weight.copy_(out_weight)
            self.attention.outLinear.weight.copy_(out_weight)
            # self.attn.query.bias.copy_(query_bias)
            self.attention.queryLinear.bias.copy_(query_bias)
            # self.attn.key.bias.copy_(key_bias)
            self.attention.keyLinear.bias.copy_(key_bias)
            # self.attn.value.bias.copy_(value_bias)
            self.attention.valueLinear.bias.copy_(value_bias)
            # self.attn.out.bias.copy_(out_bias)
            self.attention.outLinear.bias.copy_(out_bias)
            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            # self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.mlp.fc1.weight.copy_(mlp_weight_0)
            # self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.mlp.fc2.weight.copy_(mlp_weight_1)
            # self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.mlp.fc1.bias.copy_(mlp_bias_0)
            # self.ffn.fc2.bias.copy_(mlp_bias_1)
            self.mlp.fc2.bias.copy_(mlp_bias_1)

            # self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.att_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            # self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.att_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            # self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.mlp_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            # self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))
            self.mlp_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop, proj_drop):
        super(MultiHeadAttention, self).__init__()
        head_size = dim // num_heads
        self.head_size = head_size
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.softmax = Softmax(dim=-1)
        self.attn_drop = Dropout(attn_drop)
        self.proj_drop = Dropout(proj_drop)

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.normal_(self.qkv.bias, std=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).view(B, N, 3, -1, self.head_size).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]
        att_map = torch.matmul(q, k.transpose(-1, -2))
        att_map = att_map / math.sqrt(self.head_size)
        att_prob = self.softmax(att_map)
        att_prob = self.attn_drop(att_prob)
        result = torch.matmul(att_prob, v)
        result = result.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        result = self.proj(result)
        result = self.proj_drop(result)
        return result


class Mlp_self(nn.Module):
    def __init__(self, dim, dim_scale, proj_drop):
        super(Mlp_self, self).__init__()
        self.fc1 = nn.Linear(dim, dim * dim_scale)
        self.fc2 = nn.Linear(dim * dim_scale, dim)
        self.gelu = nn.GELU()
        self.dropout = Dropout(proj_drop)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        result = self.dropout(x)
        return result


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


class MSA_Module(nn.Module):
    def __init__(self, dim, num_heads, dim_scale, attn_drop, proj_drop, down_ratio=4):
        super(MSA_Module, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 3, 2, 1, output_padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        if down_ratio == 4:
            self.downBlock = nn.Sequential(
                self.down,
                self.down
            )
            self.upBlock = nn.Sequential(
                self.up,
                self.up
            )
        elif down_ratio == 2:
            self.downBlock = nn.Sequential(
                self.down
            )
            self.upBlock = nn.Sequential(
                self.up
            )
        else:
            self.downBlock = None
            self.upBlock = None
        self.attention = MultiHeadAttention(dim, num_heads, attn_drop, proj_drop)
        self.attn_norm = LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp_self(dim, dim_scale, proj_drop)
        self.mlp_norm = LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.downBlock is not None:
            x = self.downBlock(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).transpose(-1, -2).contiguous()
        temp = x
        x = self.attn_norm(x)
        x = self.attention(x)
        x += temp
        temp = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        result = x + temp
        result = result.transpose(-1, -2).contiguous().view(b, c, h, w)
        if self.upBlock is not None:
            result = self.upBlock(result)
        return result


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        return x * self.sigmoid(out)
