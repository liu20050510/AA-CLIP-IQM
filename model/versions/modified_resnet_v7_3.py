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


# ===================== 融合模块开始 =====================

# 频域注意力模块（无残差版本）
class FreqAttnPlus(nn.Module):
    def __init__(self, in_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.in_dim = in_dim
        self.mid_dim = max(in_dim // 2, 8)  # 确保最小维度

        self.reduce = nn.Conv2d(in_dim, self.mid_dim, 1)
        self.depthwise = nn.Conv2d(self.mid_dim, self.mid_dim, 3, padding=1, groups=self.mid_dim)
        self.norm = nn.BatchNorm2d(self.mid_dim)
        self.relu = nn.ReLU(inplace=True)

        # 可学习的门控机制
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.mid_dim, max(self.mid_dim // 8, 4), 1),  # 确保最小维度
            nn.ReLU(),
            nn.Conv2d(max(self.mid_dim // 8, 4), self.mid_dim, 1),
            nn.Sigmoid()
        )

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.post_proj = nn.Conv2d(self.mid_dim * 2, in_dim, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_r = self.relu(self.norm(self.depthwise(self.reduce(x))))  # B, C/2, H, W

        fft_feat = torch.fft.fft2(x_r.float(), norm='ortho')  # 复数频谱
        q = k = v = fft_feat

        # 多头重排
        q = rearrange(q, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        k = rearrange(k, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)
        v = rearrange(v, 'b (h c) h1 w1 -> b h c (h1 w1)', h=self.num_heads)

        # 注意力机制
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1) * self.temperature
        attn = F.softmax(attn.real, dim=-1)  # 使用实部计算注意力

        out = attn @ v.real
        out = rearrange(out, 'b h c (h1 w1) -> b (h c) h1 w1', h=self.num_heads, h1=H, w1=W)
        out = torch.fft.ifft2(out, norm='ortho').real  # 频域转回空间域

        # 门控频率残差
        fwm = torch.fft.ifft2(fft_feat * torch.fft.fft2(self.gate(x_r) * x_r).real)
        fwm = torch.cat([out, fwm], dim=1)

        return self.post_proj(fwm)  # 无残差连接


# 简单门控结构
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // ratio, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_channels // ratio, 4), in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


# 融合增强块 - 结合了频域注意力
class FusedEnhancedFreqBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, dilations=[1, 3, 5], num_heads=8):
        super().__init__()
        self.dw_channel = DW_Expand * c

        # 多分支卷积层
        self.conv1 = nn.Conv2d(c, self.dw_channel, kernel_size=1)
        self.branches = nn.ModuleList([
            nn.Conv2d(self.dw_channel, self.dw_channel, 3, padding=d, dilation=d, groups=self.dw_channel)
            for d in dilations
        ])

        # 门控和注意力机制
        self.sg = SimpleGate()
        self.ca = ChannelAttention(self.dw_channel // 2)
        self.conv2 = nn.Conv2d(self.dw_channel // 2, c, kernel_size=1)

        # 归一化层
        self.norm1 = nn.BatchNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c)

        # 频域注意力模块
        self.freq_attn = FreqAttnPlus(c, num_heads=num_heads)

        # 可学习参数
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        # 分支1: 多尺度空间特征提取
        y = x
        x = self.norm1(x)
        x = self.conv1(x)

        # 多分支空洞卷积融合
        z = sum([branch(x) for branch in self.branches])
        z = self.sg(z)  # 门控激活
        z = self.ca(z)  # 通道注意力
        z = self.conv2(z)  # 投影回原始通道

        # 残差连接
        y = y + self.beta * z

        # 分支2: 频域注意力增强
        x_freq = self.freq_attn(self.norm2(y))
        out = y + self.gamma * x_freq

        return out


# ===================== 融合模块结束 =====================


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


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # the 3-layer stem
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

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        # 使用FusedEnhancedFreqBlock替换原来的Bottleneck
        layers = [FusedEnhancedFreqBlock(c=self._inplanes, DW_Expand=2, dilations=[1, 3, 5])]

        # 更新通道数
        self._inplanes = planes * 4  # 保持与原始expansion=4一致

        # 添加剩余的块
        for _ in range(1, blocks):
            layers.append(FusedEnhancedFreqBlock(c=self._inplanes, DW_Expand=2, dilations=[1, 3, 5]))

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
                # 更新参数初始化以匹配新模块
                if "gamma" in name or "beta" in name:
                    nn.init.zeros_(param)
                elif "weight" in name and "conv" in name:
                    if "conv1" in name or "conv2" in name:
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        nn.init.normal_(param, std=0.01)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
                # 初始化批归一化层
                elif "norm" in name and "weight" in name:
                    if "norm1" in name or "norm2" in name:
                        nn.init.constant_(param, 1.0)
                elif "norm" in name and "bias" in name:
                    nn.init.constant_(param, 0.0)

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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x