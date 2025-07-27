# 直接引入魔改后的模型

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange  # 补充必要的导入
import math  # 补充数学函数库


# 补充缺失的FrozenBatchNorm2d类定义
class FrozenBatchNorm2d(nn.Module):
    """
    冻结的BatchNorm2d，不更新均值和方差，仅使用训练阶段预计算的统计量
    """

    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.running_mean = nn.Parameter(torch.zeros(num_features))
        self.running_var = nn.Parameter(torch.ones(num_features))
        self.training = False  # 冻结状态下始终为False

    def forward(self, x):
        # 计算标准差（避免除以零）
        std = torch.sqrt(self.running_var + self.eps)
        # 标准化操作：(x - mean) / std * weight + bias
        x = (x - self.running_mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


# 定义 MSC_Modified 模块
class MSC_Modified(nn.Module):
    def __init__(self, dim, num_heads=8, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0.):
        super(MSC_Modified, self).__init__()
        # 增加参数合法性检查
        assert len(kernel) == 3 and len(s) == 3 and len(pad) == 3, "kernel, s, pad must be length 3"

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        # 可学习 TopK 比率参数（限制范围避免极端值）
        self.k_ratio1 = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.k_ratio2 = nn.Parameter(torch.tensor(0.25), requires_grad=True)

        # 可变卷积代替池化（深度可分离卷积）
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel[0], stride=s[0], padding=pad[0], groups=dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel[1], stride=s[1], padding=pad[1], groups=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=kernel[2], stride=s[2], padding=pad[2], groups=dim)

        self.layer_norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        B, C, H, W = y.shape  # y: 多尺度卷积输入特征图

        # 多尺度卷积 + 双线性插值融合
        y1 = self.conv1(y)
        y2 = self.conv2(y)
        y3 = self.conv3(y)

        # 将不同尺度特征上采样到同一大小后加和（以y1为基准）
        y2 = F.interpolate(y2, size=y1.shape[2:], mode='bilinear', align_corners=False)
        y3 = F.interpolate(y3, size=y1.shape[2:], mode='bilinear', align_corners=False)
        y = y1 + y2 + y3  # 多尺度特征融合

        # 特征图转序列并归一化
        y = rearrange(y, 'b c h w -> b (h w) c')  # (B, H1*W1, C)
        y = self.layer_norm(y)

        # x处理：特征图转序列
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, H*W, C)
        B, N, C = x.shape  # N = H*W
        N1 = y.shape[1]  # N1 = H1*W1（融合后的序列长度）

        # 计算K和V（来自多尺度特征y）
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # k: (B, num_heads, N1, head_dim), v: (B, num_heads, N1, head_dim)

        # 计算Q（来自输入特征x）
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1,
                                                                                 3)  # (B, num_heads, N, head_dim)

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N1)

        # 动态TopK计算（通过sigmoid限制比率在0-1之间）
        k1_ratio = self.sigmoid(self.k_ratio1)  # 确保比率在0-1
        k2_ratio = self.sigmoid(self.k_ratio2)
        k1_num = (N1 * k1_ratio).int().clamp(1, N1)  # 限制最少1个，最多N1个
        k2_num = (N1 * k2_ratio).int().clamp(1, N1)

        # TopK掩码函数（提取重要注意力）
        def topk_mask(attn_map, k_num):
            mask = torch.zeros_like(attn_map)
            # 获取每个query的top-k key索引
            topk_indices = torch.topk(attn_map, k=k_num.item(), dim=-1, largest=True)[1]
            mask.scatter_(-1, topk_indices, 1.0)  # 标记top-k位置
            # 仅保留top-k的注意力分数，其余设为负无穷（softmax后接近0）
            attn_masked = torch.where(mask > 0, attn_map, torch.full_like(attn_map, float('-inf')))
            attn_masked = F.softmax(attn_masked, dim=-1)
            attn_masked = self.attn_drop(attn_masked)
            return attn_masked

        # 计算两种TopK注意力
        attn1 = topk_mask(attn, k1_num)
        attn2 = topk_mask(attn, k2_num)

        # 注意力加权求和
        out1 = attn1 @ v  # (B, num_heads, N, head_dim)
        out2 = attn2 @ v

        # 加权融合两种输出
        out = 0.6 * out1 + 0.4 * out2  # 权重可根据任务调整

        # 输出转换回特征图格式
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.proj(out)  # 线性投影
        out = self.proj_drop(out)
        out = out + x  # 残差连接

        # 序列转特征图（通过平方根计算空间尺寸）
        hw = int(math.sqrt(N))  # 确保N是平方数（特征图尺寸为hw x hw）
        out = rearrange(out, 'b (h w) c -> b c h w', h=hw, w=hw)

        return out


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    将模块中所有BatchNorm2d/SyncBatchNorm转换为FrozenBatchNorm2d
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
        # 替换为冻结版本
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data.clone()
        res.running_var.data = module.running_var.data.clone()
        res.eps = module.eps
    else:
        # 递归处理子模块
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


class Bottleneck(nn.Module):
    expansion = 4  # 瓶颈结构的通道扩展倍数

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # 卷积层（所有卷积步长为1，步长>1时通过avgpool实现下采样）
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)  # 1x1卷积降维
        self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)  # 3x3卷积
        self.bn2 = nn.BatchNorm2d(planes)
        self.act2 = nn.ReLU(inplace=True)

        # 步长>1时通过平均池化下采样
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)  # 1x1卷积升维
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        # 当步长>1或输入输出通道不匹配时，需要下采样模块
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),  # 平均池化下采样
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),  # 1x1卷积调整通道
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x  # 残差连接

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)  # 下采样（若需要）
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)  # 调整残差分支的尺寸和通道

        out += identity  # 残差相加
        out = self.act3(out)
        return out


class AttentionPool2d(nn.Module):
    """
    基于注意力的池化层，用于将特征图转换为全局特征
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # 位置嵌入（+1是为了包含全局均值特征）
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # 注意力投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)  # 输出投影
        self.num_heads = num_heads

    def forward(self, x):
        # 特征图转序列：NCHW -> (HW)NC
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)
        # 拼接全局均值特征（作为"类令牌"）
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # 添加位置嵌入
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        # 多头注意力计算
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

        return x[0]  # 返回全局特征（类令牌对应的输出）


class ModifiedResNet(nn.Module):
    """
    改进的ResNet：
    - 3层stem卷积（替代原1层）
    - 抗锯齿下采样（步长>1时先平均池化）
    - 注意力池化（替代全局平均池化）
    - 集成MSC_Modified模块增强特征交互
    """

    def __init__(self, layers, output_dim, heads, image_size=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.image_size = image_size

        # 3层stem卷积（逐步提升通道数）
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.act3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)  # stem后的下采样

        # 残差块组（layers控制每个组的块数量）
        self._inplanes = width  # 当前输入通道数（动态更新）
        self.layer1 = self._make_layer(width, layers[0])  # 不改变尺寸
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)  # 下采样x2
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)  # 下采样x2
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)  # 下采样x2

        # 计算最终特征图通道数（width*8 * 4 = width*32）
        embed_dim = width * 32
        # 注意力池化层（将特征图转为全局特征）
        self.attnpool = AttentionPool2d(image_size // 32, embed_dim, heads, output_dim)

        # 初始化MSC模块（使用最终特征图通道数作为dim）
        self.msc_module = MSC_Modified(dim=embed_dim, num_heads=heads)

        # 初始化参数
        self.init_parameters()

    def _make_layer(self, planes, blocks, stride=1):
        """构建残差块组"""
        layers = [Bottleneck(self._inplanes, planes, stride)]  # 第一个块可能包含下采样

        self._inplanes = planes * Bottleneck.expansion  # 更新输入通道数
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))  # 后续块不改变尺寸

        return nn.Sequential(*layers)

    def init_parameters(self):
        """参数初始化"""
        if self.attnpool is not None:
            # 注意力层参数初始化（正态分布）
            std = self.attnpool.c_proj.in_features ** -0.5
            nn.init.normal_(self.attnpool.q_proj.weight, std=std)
            nn.init.normal_(self.attnpool.k_proj.weight, std=std)
            nn.init.normal_(self.attnpool.v_proj.weight, std=std)
            nn.init.normal_(self.attnpool.c_proj.weight, std=std)

        # 残差块中最后一个BN层初始化为0（类似ResNet的初始化）
        for resnet_block in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in resnet_block.named_parameters():
                if name.endswith("bn3.weight"):
                    nn.init.zeros_(param)

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        """冻结模型参数（用于迁移学习）"""
        assert unlocked_groups == 0, '当前版本不支持部分解锁'
        for param in self.parameters():
            param.requires_grad = False
        if freeze_bn_stats:
            freeze_batch_norm_2d(self)  # 冻结BN统计量

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        """预留梯度检查点接口（用于节省显存）"""
        pass

    def stem(self, x):
        """stem卷积流程"""
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    def forward(self, x):
        """前向传播流程"""
        x = self.stem(x)  # stem处理
        x = self.layer1(x)  # 第一层残差块
        x = self.layer2(x)  # 第二层残差块（下采样）
        x = self.layer3(x)  # 第三层残差块（下采样）
        x = self.layer4(x)  # 第四层残差块（下采样）

        x = self.msc_module(x, x)  # 调用MSC模块（x同时作为查询和卷积输入）

        x = self.attnpool(x)  # 注意力池化得到全局特征

        return x