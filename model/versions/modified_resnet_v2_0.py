# version2.0 引入多尺度特征融合、空洞卷积以及改进的注意力池化层

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FrozenBatchNorm2d

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

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        # 引入空洞卷积，调整dilation参数以适应织物瑕疵检测
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=dilation, dilation=dilation, bias=False)
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


class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # resnets have only one self-attention, norm and larger scale perform better
        if self.settings == 'resnet':
            k = k / (k.norm(p=2, dim=-1, keepdim=True) + 1e-6)
            q = k
            scale = self.scale * 8
        else:
            scale = self.scale

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        #x = (attn @ v).transpose(1, 2).reshape(B, N, C) # clip_surgery
        x = v.transpose(1, 2).reshape(B, N, C) # mask_clip
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]


class FabricAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 16, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self.attn = None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        # 引入额外的注意力层
        self.extra_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        # reform transformer layer after init and load weights, using v only
        if self.attn == None:
            self.attn = Attention(self.output_dim, self.embed_dim, self.num_heads, True)
            self.attn.qkv.weight = torch.nn.Parameter(torch.cat([self.v_proj.weight, self.v_proj.weight, self.v_proj.weight], 0))
            self.attn.qkv.bias = torch.nn.Parameter(torch.cat([self.v_proj.bias, self.v_proj.bias, self.v_proj.bias]))
            self.attn.proj.weight = self.c_proj.weight
            self.attn.proj.bias = self.c_proj.bias

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[0] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        # 额外的注意力操作
        x = x.transpose(0, 1)
        x, _ = self.extra_attn(x, x, x)
        x = x.transpose(0, 1)

        x, x_ori = self.attn(x.transpose(0, 1))

        # cls token from the original path, and img tokens from the new path
        x[:, 0, :] = x_ori[:, 0, :]
        return x


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
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2, dilation=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2, dilation=4)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        # 增加织物特定的注意力机制
        self.fabric_attention = FabricAttention(embed_dim)

        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        layers = [Bottleneck(self._inplanes, planes, stride, dilation)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, dilation=dilation))

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
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # 多尺度特征融合，采用更复杂的融合策略
        fused_features = self.fuse_multi_scale_features([x1, x2, x3, x4])

        # 应用织物特定的注意力机制
        fused_features = self.fabric_attention(fused_features)

        x = self.attnpool(fused_features)

        return x

    def fuse_multi_scale_features(self, features):
        # 对不同尺度的特征进行卷积调整
        adjusted_features = []
        for i, feature in enumerate(features):
            in_channels = feature.size(1)
            out_channels = features[-1].size(1)
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1).to(feature.device)
            adjusted_feature = conv(feature)
            adjusted_features.append(adjusted_feature)

        # 上采样到相同尺寸
        target_size = adjusted_features[-1].size()[2:]
        resized_features = []
        for feature in adjusted_features:
            resized_feature = F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
            resized_features.append(resized_feature)

        # 加权融合
        weights = nn.Parameter(torch.ones(len(resized_features), requires_grad=True)).softmax(dim=0)
        weighted_features = [w * f for w, f in zip(weights, resized_features)]
        fused = sum(weighted_features)

        return fused