# 插入到每个残差层之后

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FrozenBatchNorm2d
from einops import rearrange


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


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


# 高效门控结构 (优化版)
class EfficientGate(nn.Module):
    def __init__(self, reduction=4):
        super().__init__()
        self.reduction = reduction
        self.compression = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        ) if reduction > 1 else nn.Identity()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        if self.reduction > 1:
            # 双池化压缩
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


# 魔改增强模块：IndustrialFusionBlock (改进版)
class IndustrialFusionBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1, 3, 5], reduction=4):
        super().__init__()
        self.dw_channel = DW_Expand * c

        # 多尺度特征提取
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1)

        # 多分支卷积（不同空洞率）
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dw_channel, self.dw_channel, 3,
                          padding=d, dilation=d, groups=self.dw_channel),
                nn.GELU(),
                nn.Conv2d(self.dw_channel, self.dw_channel, 1)
            ) for d in dilations
        ])

        # 门控和注意力
        self.gate = EfficientGate(reduction)
        self.attn = EnhancedAttention(self.dw_channel // 2, ratio=reduction)

        # 输出转换
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dw_channel // 2, c, kernel_size=1),
            nn.GELU()
        )

        # 归一化层
        self.norm1 = nn.GroupNorm(4, c)
        self.norm2 = nn.GroupNorm(4, c)

        # 使用FreqAttnPlus替换原频域模块
        self.freq = FreqAttnPlus(c, num_heads=min(8, c // 2))

        # 可学习参数
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

        # 残差连接
        self.res_conv = nn.Conv2d(c, c, 1) if c != self.dw_channel // 2 else nn.Identity()

    def forward(self, x):
        residual = x

        # 分支1：空间域处理
        x = self.norm1(x)
        x = self.conv1(x)

        # 多分支特征融合
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        x = sum(branch_outputs)

        # 门控机制
        x = self.gate(x)
        x = self.attn(x)
        x = self.conv2(x)

        # 残差连接
        x = residual + self.beta * x

        # 分支2：频域处理
        y = self.norm2(x)
        freq_out = self.freq(y)

        # 特征融合 (直接相加)
        out = x + self.gamma * freq_out
        return out


class ModifiedResNet(nn.Module):
    """
    改进的ResNet，整合了工业质检增强模块，提升对细微缺陷的检测能力
    - 保留原有的stem和基础残差结构
    - 在每个残差层后添加IndustrialFusionBlock增强模块
    - 结合频域和空域注意力机制，提高对微小缺陷的敏感度
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # 残差层定义（新增工业增强模块）
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.fusion1 = IndustrialFusionBlock(c=width * 4)  # 对应layer1输出维度（width*4）

        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.fusion2 = IndustrialFusionBlock(c=width * 8)  # 对应layer2输出维度（width*8）

        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.fusion3 = IndustrialFusionBlock(c=width * 16)  # 对应layer3输出维度（width*16）

        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.fusion4 = IndustrialFusionBlock(c=width * 32)  # 对应layer4输出维度（width*32）

        # 注意力池化保持不变
        embed_dim = width * 32  # ResNet特征维度
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def init_parameters(self):
        if self.attnpool is not None:
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

        # 初始化工业增强模块参数
        for fusion_block in [self.fusion1, self.fusion2, self.fusion3, self.fusion4]:
            for name, param in fusion_block.named_parameters():
                if 'conv' in name and 'weight' in name:
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bn' in name or 'norm' in name:
                    if 'weight' in name:
                        nn.init.ones_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        # FIXME support for non-transformer
        pass

    def stem(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        x = self.stem(x)

        # 残差层+工业增强模块
        x = self.layer1(x)
        x = self.fusion1(x)  # 增强layer1输出

        x = self.layer2(x)
        x = self.fusion2(x)  # 增强layer2输出

        x = self.layer3(x)
        x = self.fusion3(x)  # 增强layer3输出

        x = self.layer4(x)
        x = self.fusion4(x)  # 增强layer4输出

        x = self.attnpool(x)
        return x