from torch import nn
import ipdb
import math


class SimpleAdapter(nn.Module):
    def __init__(self, c_in, c_out=768):
        super(SimpleAdapter, self).__init__()
        self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())

    def forward(self, x):
        x = self.fc(x)
        return x


class SimpleProj(nn.Module):
    def __init__(self, c_in, c_out=768, relu=True):
        super(SimpleProj, self).__init__()
        if relu:
            self.fc = nn.Sequential(nn.Linear(c_in, c_out, bias=False), nn.LeakyReLU())
        else:
            self.fc = nn.Linear(c_in, c_out, bias=False)

    def forward(self, x):
        x = self.fc(x)
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FreqAttnPlus(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super(FreqAttnPlus, self).__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.mid_dim = in_dim // 2

        self.reduce = nn.Conv2d(in_dim, self.mid_dim, 1)
        self.depthwise = nn.Conv2d(self.mid_dim, self.mid_dim, 3, padding=1, groups=self.mid_dim)
        self.norm = nn.BatchNorm2d(self.mid_dim)
        self.relu = nn.ReLU(inplace=True)

        # learnable gate
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.mid_dim, self.mid_dim // 8, 1),
            nn.ReLU(),
            nn.Conv2d(self.mid_dim // 8, self.mid_dim, 1),
            nn.Sigmoid()
        )

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.post_proj = nn.Conv2d(self.mid_dim * 2, in_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_r = self.relu(self.norm(self.depthwise(self.reduce(x))))  # B, C/2, H, W

        fft_feat = torch.fft.fft2(x_r.float(), norm='ortho')  # complex
        q = k = v = fft_feat

        # reshape into heads
        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = F.softmax(attn.real, dim=-1)  # 只用实部参与 softmax

        out = attn @ v.real
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads, h1=H, w1=W)
        out = torch.fft.ifft2(out, norm='ortho').real  # 频域回到空间

        # 残差增强路径（门控频率残差）
        fwm = torch.fft.ifft2(fft_feat * torch.fft.fft2(self.gate(x_r) * x_r)).real
        fwm = torch.cat([out, fwm], dim=1)

        out = self.post_proj(fwm) + x  # 加残差
        return out


# 高效门控结构 (优化版) - 修正通道数问题
class EfficientGate(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.reduction = reduction
        self.in_channels = in_channels
        self.out_channels = self.in_channels // reduction

        # 压缩通道 + 扩展通道
        if reduction > 1:
            self.compression = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.Sigmoid(),
                # 新增：将通道数扩展回原始分割后的大小（如1024）
                nn.Conv2d(self.out_channels, self.in_channels, kernel_size=1, bias=False)
            )
        else:
            self.compression = nn.Identity()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)

        if self.reduction > 1:
            compressed = self.compression(x1)
            return compressed * x2
        return x1 * x2


# 增强型通道空间注意力
class EnhancedAttention(nn.Module):
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super().__init__()

        # 通道注意力
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

        # 残差连接权重
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 通道注意力
        ca = self.channel_att(x)
        ca_out = x * ca

        # 空间注意力
        sa_max, _ = torch.max(x, dim=1, keepdim=True)
        sa_avg = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([sa_max, sa_avg], dim=1))
        sa_out = x * sa

        # 注意力融合
        return x + self.gamma * (ca_out + sa_out)


# 魔改增强模块：IndustrialFusionBlock (改进版) - 优化通道处理
class IndustrialFusionBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1, 3, 5], reduction=4):
        super().__init__()
        self.dw_channel = DW_Expand * c  # 默认为2048通道

        # 多尺度特征提取
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1)  # 1024->2048

        # 多分支卷积（不同空洞率）
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dw_channel, self.dw_channel, 3,
                          padding=d, dilation=d, groups=self.dw_channel),
                nn.GELU(),
                nn.Conv2d(self.dw_channel, self.dw_channel, 1)
            ) for d in dilations
        ])

        # 门控和注意力 - 使用修正后的EfficientGate
        self.gate = EfficientGate(in_channels=self.dw_channel // 2, reduction=reduction)
        self.attn = EnhancedAttention(self.dw_channel // 2, ratio=reduction)

        # 输出转换
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dw_channel // 2, c, kernel_size=1),  # 1024->1024
            nn.GELU()
        )

        # 归一化层
        self.norm1 = nn.GroupNorm(4, c)
        self.norm2 = nn.GroupNorm(4, c)

        # 使用FreqAttnPlus替换原频域模块
        self.freq = FreqAttnPlus(c, num_heads=min(8, c // 2))  # 保持1024通道

        # 可学习参数
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

        # 残差连接
        self.res_conv = nn.Conv2d(c, c, 1) if c != self.dw_channel // 2 else nn.Identity()

    def forward(self, x):
        residual = x  # [B, 1024, H, W]

        # 分支1：空间域处理
        x = self.norm1(x)
        x = self.conv1(x)  # [B, 2048, H, W]

        # 多分支特征融合
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        x = sum(branch_outputs)  # [B, 2048, H, W]

        # 门控机制 - 现在通道数正确匹配
        x = self.gate(x)  # [B, 2048, H, W]
        x = self.attn(x)  # [B, 2048, H, W]
        x = self.conv2(x)  # [B, 1024, H, W]

        # 残差连接
        x = residual + self.beta * x  # [B, 1024, H, W]

        # 分支2：频域处理
        y = self.norm2(x)  # [B, 1024, H, W]
        freq_out = self.freq(y)  # [B, 1024, H, W]

        # 特征融合 (直接相加)
        out = x + self.gamma * freq_out  # [B, 1024, H, W]
        return out