import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj


# 交叉模态注意力模块
class CrossModelAtt(nn.Module):
    def __init__(self, feature_dim, height, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))  # 可学习的参数
        self.softmax = nn.Softmax(dim=-1)  # Softmax 用于归一化注意力权重
        self.height = height
        self.width = width

    def forward(self, img_feat, text_feat):
        # img_feat: [B, C, H, W]
        # text_feat: [B, C, H, W]
        B, C, H, W = img_feat.shape

        # 特征图展平
        q = img_feat.view(B, C, -1)  # [B, C, H*W]
        k = text_feat.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]

        # 计算注意力感知矩阵
        attention_map = torch.bmm(q, k)  # [B, C, C]
        attention_map = self.softmax(attention_map)  # [B, C, C]

        # 融合
        v = text_feat.view(B, C, -1)  # [B, C, H*W]
        attention_info = torch.bmm(attention_map, v)  # [B, C, H*W]

        # 重构为原始的H和W维度
        attention_info = attention_info.view(B, C, H, W)

        # 加权和原特征图
        output = self.gamma * attention_info + img_feat  # 加权融合后的结果

        return output


class AdaptedCLIP(nn.Module):
    def __init__(
            self,
            clip_model,
            text_adapt_weight: float = 0.1,
            image_adapt_weight: float = 0.1,
            text_adapt_until: int = 3,
            image_adapt_until: int = 6,
            levels: list = [6, 12, 18, 24],
            relu: bool = True,
            cross_attn_height: int = 16,  # 交叉注意力特征图高度
            cross_attn_width: int = 16,  # 交叉注意力特征图宽度
            **kwargs,
    ):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.text_adapt_until = text_adapt_until
        self.image_adapt_until = image_adapt_until
        self.t_w = text_adapt_weight
        self.i_w = image_adapt_weight
        self.levels = levels

        # 图像适配器
        layer_adapters = nn.ModuleList(
            [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
        )
        seg_proj = nn.ModuleList(
            [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
        )
        det_proj = SimpleProj(1024, 768, relu)

        # 新增：交叉模态注意力模块
        self.cross_attn = CrossModelAtt(
            feature_dim=1024,
            height=cross_attn_height,
            width=cross_attn_width
        )

        self.image_adapter = nn.ModuleDict(
            {
                "layer_adapters": layer_adapters,
                "seg_proj": seg_proj,
                "det_proj": det_proj,
            }
        )

        # 文本适配器
        self.text_adapter = nn.ModuleList(
            [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
            + [SimpleProj(768, 768, relu=True)]
        )

        # 默认文本特征（用于交叉注意力）
        self.default_text_features = None
        self._init_weights_()

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 初始化交叉注意力参数
        if hasattr(self, 'cross_attn'):
            nn.init.zeros_(self.cross_attn.gamma)

    def forward_original(self, x, modality="visual"):
        if modality == "visual":
            cls_features, patch_features = self.clipmodel.encode_image(x, [24])
            patch_features = [
                self.clipmodel.visual._global_pool(t)[1] for t in patch_features
            ]
            patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
            patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
            return patch_features, cls_features
        else:
            raise ValueError("modality must be visual")

    def set_default_text_features(self, text_features):
        """设置默认文本特征，用于交叉注意力"""
        self.default_text_features = text_features

    def forward(self, x, text_feat=None):
        # 图像特征提取
        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.image_encoder.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        tokens = []
        for i in range(24):
            x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if i < self.image_adapt_until:
                adapt_out = self.image_adapter["layer_adapters"][i](x)
                adapt_out = (
                        adapt_out
                        * x.norm(dim=-1, keepdim=True)
                        / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.i_w * adapt_out + (1 - self.i_w) * x
            if i + 1 in self.levels:
                tokens.append(x[1:, :, :])

        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]

        # 使用交叉模态注意力（强制使用）
        # 如果没有显式提供text_feat，则使用默认的文本特征
        if text_feat is None:
            text_feat = self.default_text_features

            # 如果有文本特征，则应用交叉模态注意力
            if text_feat is not None:
                # 调整文本特征形状以匹配图像特征
                if len(text_feat.shape) == 2:
                    # 如果text_feat是2维的[B, C]，扩展为3维[B, C, 1]
                    text_feat = text_feat.unsqueeze(-1)
                B, C, _ = text_feat.shape
                H, W = self.cross_attn.height, self.cross_attn.width

                # 将文本特征扩展为空间特征图
                text_feat_reshaped = text_feat.unsqueeze(-1).repeat(1, 1, H * W)  # [B, C, H*W]
                text_feat_reshaped = text_feat_reshaped.view(B, C, H, W)  # [B, C, H, W]

                # 对每个层级的图像特征应用交叉注意力
                fused_tokens = []
                for token in tokens:
                    B, L, C = token.shape
                    # 动态计算H和W，而不是使用固定的值
                    HW = int(L ** 0.5)
                    token_reshaped = token.permute(0, 2, 1).view(B, C, HW, HW)  # [B, C, HW, HW]
                    # 调整text_feat_reshaped的大小以匹配token的大小
                    text_feat_resized = F.interpolate(text_feat_reshaped, size=(HW, HW), mode='bilinear',
                                                      align_corners=False)
                    fused_token = self.cross_attn(token_reshaped, text_feat_resized)  # [B, C, HW, HW]
                    fused_token = fused_token.view(B, C, L).permute(0, 2, 1)  # [B, L, C]
                    fused_tokens.append(fused_token)
                tokens = fused_tokens
        # 投影到输出维度
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)
        return seg_tokens, det_token

    def encode_text(self, text, adapt_text=True):
        if not adapt_text:
            return self.clipmodel.encode_text(text)
        cast_dtype = self.clipmodel.transformer.get_cast_dtype()
        x = self.clipmodel.token_embedding(text).to(
            cast_dtype
        )  # [batch_size, n_ctx, d_model]

        x = x + self.clipmodel.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        for i in range(12):
            x, attn = self.clipmodel.transformer.resblocks[i](
                x, attn_mask=self.clipmodel.attn_mask
            )
            if i < self.text_adapt_until:
                adapt_out = self.text_adapter[i](x)
                adapt_out = (
                        adapt_out
                        * x.norm(dim=-1, keepdim=True)
                        / adapt_out.norm(dim=-1, keepdim=True)
                )
                x = self.t_w * adapt_out + (1 - self.t_w) * x
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # 应用最后的投影
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        return x

# import torch
# from torch import nn
# import torch.nn.functional as F
# from .adapter_modules import SimpleAdapter, SimpleProj
#
#
# # 导入LightBiMAFusion模块
# class LightBiMAFusion(nn.Module):
#     """
#     LightBiMAFusion: 轻量化双向模态融合模块
#     - 高分辨率输入支持（如 256×256）
#     - 注意力仅在低分辨率上计算，节省显存
#     - 支持图像语义双向融合
#     """
#
#     def __init__(self, img_channels, txt_channels, mid_channels=64, attn_size=64):
#         super(LightBiMAFusion, self).__init__()
#         self.attn_size = attn_size
#
#         self.img_proj = nn.Conv2d(img_channels, mid_channels, kernel_size=1)
#         self.txt_proj = nn.Conv2d(txt_channels, mid_channels, kernel_size=1)
#
#         self.gamma_img2txt = nn.Parameter(torch.zeros(1))
#         self.gamma_txt2img = nn.Parameter(torch.zeros(1))
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, img_feat, txt_feat):
#         B, _, H, W = img_feat.size()
#         target_size = (H, W)
#
#         # 计算池化参数以替代自适应池化
#         stride_h = H // self.attn_size
#         kernel_h = H - (self.attn_size - 1) * stride_h
#         stride_w = W // self.attn_size
#         kernel_w = W - (self.attn_size - 1) * stride_w
#
#         # 使用自适应池化到目标注意力尺寸
#         img_down = F.adaptive_avg_pool2d(img_feat, (self.attn_size, self.attn_size))
#         txt_down = F.adaptive_avg_pool2d(txt_feat, (self.attn_size, self.attn_size))
#
#         # 通道映射
#         img_proj = self.img_proj(img_down)
#         txt_proj = self.txt_proj(txt_down)
#
#         B, C, h, w = img_proj.shape
#         N = h * w
#
#         # reshape
#         Q_txt = txt_proj.view(B, C, N)
#         K_img = img_proj.view(B, C, N)
#         V_img = img_proj.view(B, C, N).permute(0, 2, 1)
#         V_txt = txt_proj.view(B, C, N).permute(0, 2, 1)
#
#         # 注意力矩阵
#         attn_img2txt = self.softmax(torch.bmm(Q_txt.permute(0, 2, 1), K_img))
#         attn_txt2img = self.softmax(torch.bmm(K_img.permute(0, 2, 1), Q_txt))
#
#         # 注意力融合
#         fusion_img = torch.bmm(attn_img2txt, V_img).permute(0, 2, 1).view(B, C, h, w)
#         fusion_txt = torch.bmm(attn_txt2img, V_txt).permute(0, 2, 1).view(B, C, h, w)
#
#         # 上采样恢复
#         fusion_img_up = F.interpolate(fusion_img, size=target_size, mode='bilinear', align_corners=False)
#         fusion_txt_up = F.interpolate(fusion_txt, size=target_size, mode='bilinear', align_corners=False)
#
#         # 残差融合
#         out_img = self.gamma_img2txt * fusion_img_up + self.txt_proj(txt_feat)
#         out_txt = self.gamma_txt2img * fusion_txt_up + self.img_proj(img_feat)
#
#         return out_img, out_txt
#
#
# class AdaptedCLIP(nn.Module):
#     def __init__(
#             self,
#             clip_model,
#             text_adapt_weight: float = 0.1,
#             image_adapt_weight: float = 0.1,
#             text_adapt_until: int = 3,
#             image_adapt_until: int = 6,
#             levels: list = [6, 12, 18, 24],
#             relu: bool = True,
#             fusion_mid_channels: int = 64,
#             fusion_attn_size: int = 64,
#             **kwargs,
#     ):
#         super().__init__()
#         self.clipmodel = clip_model
#         self.image_encoder = clip_model.visual
#         self.text_adapt_until = text_adapt_until
#         self.image_adapt_until = image_adapt_until
#         self.t_w = text_adapt_weight
#         self.i_w = image_adapt_weight
#         self.levels = levels
#
#         # 图像适配器
#         layer_adapters = nn.ModuleList(
#             [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
#         )
#         seg_proj = nn.ModuleList(
#             [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
#         )
#         det_proj = SimpleProj(1024, 768, relu)
#         self.image_adapter = nn.ModuleDict(
#             {
#                 "layer_adapters": layer_adapters,
#                 "seg_proj": seg_proj,
#                 "det_proj": det_proj,
#             }
#         )
#
#         # 文本适配器
#         self.text_adapter = nn.ModuleList(
#             [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
#             + [SimpleProj(768, 768, relu=True)]
#         )
#
#         # 初始化双向模态融合模块
#         self.bi_fusion = LightBiMAFusion(
#             img_channels=1024,
#             txt_channels=768,
#             mid_channels=fusion_mid_channels,
#             attn_size=fusion_attn_size
#         )
#
#         # 添加融合特征投影层
#         self.fusion_proj = nn.Conv2d(fusion_mid_channels, 1024, kernel_size=1)
#
#         self._init_weights_()
#
#     def _init_weights_(self):
#         for p in self.image_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for p in self.text_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         # 初始化融合模块参数
#         for p in self.bi_fusion.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         # 初始化融合投影层
#         nn.init.xavier_uniform_(self.fusion_proj.weight)
#         if self.fusion_proj.bias is not None:
#             nn.init.zeros_(self.fusion_proj.bias)
#
#     def forward_original(self, x, modality="visual"):
#         if modality == "visual":
#             cls_features, patch_features = self.clipmodel.encode_image(x, [24])
#             patch_features = [
#                 self.clipmodel.visual._global_pool(t)[1] for t in patch_features
#             ]
#             patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
#             patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
#             return patch_features, cls_features
#         else:
#             raise ValueError("modality must be visual")
#
#     def forward(self, x, text_feat=None):
#         # 图像特征提取
#         x = self.image_encoder.conv1(x)
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         x = x.permute(0, 2, 1)
#
#         x = torch.cat(
#             [
#                 self.image_encoder.class_embedding.to(x.dtype)
#                 + torch.zeros(
#                     x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
#                 ),
#                 x,
#             ],
#             dim=1,
#         )
#         x = x + self.image_encoder.positional_embedding.to(x.dtype)
#
#         x = self.image_encoder.patch_dropout(x)
#         x = self.image_encoder.ln_pre(x)
#
#         x = x.permute(1, 0, 2)
#
#         tokens = []
#         for i in range(24):
#             x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
#             if i < self.image_adapt_until:
#                 adapt_out = self.image_adapter["layer_adapters"][i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.i_w * adapt_out + (1 - self.i_w) * x
#             if i + 1 in self.levels:
#                 tokens.append(x[1:, :, :])
#
#         x = x.permute(1, 0, 2)
#         tokens = [t.permute(1, 0, 2) for t in tokens]
#         tokens = [self.image_encoder.ln_post(t) for t in tokens]
#
#         # 如果提供了文本特征，则进行双向模态融合
#         if text_feat is not None:
#             # 调整特征形状以适应融合模块 (B, N, C) -> (B, C, H, W)
#             B, N, C_img = tokens[-1].shape
#             H = W = int(N ** 0.5)
#             img_feat_2d = tokens[-1].permute(0, 2, 1).view(B, C_img, H, W)
#
#             # 文本特征形状调整
#             B, C_txt, _ = text_feat.shape
#             txt_feat_2d = text_feat.unsqueeze(-1).repeat(1, 1, H, W)  # 扩展到空间维度
#
#             # 双向融合
#             fused_img, fused_txt = self.bi_fusion(img_feat_2d, txt_feat_2d)
#
#             # 投影融合特征到原始通道数
#             fused_img = self.fusion_proj(fused_img)
#
#             # 恢复形状 (B, C, H, W) -> (B, N, C)
#             fused_img = fused_img.view(B, 1024, N).permute(0, 2, 1)
#
#             # 残差连接
#             tokens[-1] = tokens[-1] + fused_img
#
#         # 后续处理
#         seg_tokens = [
#             self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
#         ]
#         seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
#         det_token = self.image_adapter["det_proj"](tokens[-1])
#         det_token = F.normalize(det_token, dim=-1).mean(1)
#         return seg_tokens, det_token
#
#     def encode_text(self, text, adapt_text=True):
#         if not adapt_text:
#             return self.clipmodel.encode_text(text)
#         cast_dtype = self.clipmodel.transformer.get_cast_dtype()
#         x = self.clipmodel.token_embedding(text).to(
#             cast_dtype
#         )  # [batch_size, n_ctx, d_model]
#
#         x = x + self.clipmodel.positional_embedding.to(cast_dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#
#         for i in range(12):
#             x, attn = self.clipmodel.transformer.resblocks[i](
#                 x, attn_mask=self.clipmodel.attn_mask
#             )
#             if i < self.text_adapt_until:
#                 adapt_out = self.text_adapter[i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.t_w * adapt_out + (1 - self.t_w) * x
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#         # 文本特征适配
#         txt_feat = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
#         return txt_feat



# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import List
# from .adapter_modules import SimpleAdapter, SimpleProj
#
#
# class EfficientCovLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, k):
#         super().__init__()
#         # 深度卷积层 (分组卷积)
#         self.depthwise_convs = nn.ModuleDict({
#             'k3': nn.Conv2d(dim_in, dim_in, kernel_size=3, padding=1, groups=dim_in),
#             'k5': nn.Conv2d(dim_in, dim_in, kernel_size=5, padding=2, groups=dim_in),
#             'k7': nn.Conv2d(dim_in, dim_in, kernel_size=7, padding=3, groups=dim_in),
#             'k15': nn.Conv2d(dim_in, dim_in, kernel_size=(1, 5), padding=(0, 2), groups=dim_in),
#             'k51': nn.Conv2d(dim_in, dim_in, kernel_size=(5, 1), padding=(2, 0), groups=dim_in)
#         })
#
#         # 点卷积层 (1x1卷积)
#         self.pointwise_convs = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=1) for _ in range(k)]
#         )
#
#         # 1x1卷积分支
#         self.conv1x1 = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=1) for _ in range(k)]
#         )
#
#     def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
#         processed = []
#         for i, x in enumerate(tokens):
#             B, L, C = x.shape
#             H = W = int(L ** 0.5)
#             x_2d = x.view(B, H, W, C).permute(0, 3, 1, 2)
#
#             # 并行计算所有深度卷积
#             depthwise_features = []
#             for conv in self.depthwise_convs.values():
#                 depthwise_features.append(conv(x_2d))
#
#             # 合并多尺度特征
#             fused = torch.stack(depthwise_features).sum(dim=0)
#
#             # 点卷积处理
#             pointwise_out = self.pointwise_convs[i](fused)
#
#             # 添加1x1分支结果
#             conv1x1_out = self.conv1x1[i](x_2d)
#
#             # 特征融合
#             combined = pointwise_out + conv1x1_out
#
#             # 恢复原始形状
#             processed.append(combined.permute(0, 2, 3, 1).view(B, L, -1))
#
#         return processed
#
#
# class LinearLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, k):
#         super().__init__()
#         self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])
#
#     def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
#         return [self.fc[i](x) for i, x in enumerate(tokens)]
#
#
# class MultiScaleFeatureExtractor(nn.Module):
#     def __init__(self, dim_in, dim_out, num_layers):
#         super().__init__()
#         self.cov_layer = EfficientCovLayer(dim_in, dim_out, num_layers)
#         self.linear_layer = LinearLayer(dim_out, dim_out, num_layers)
#         self.norm = nn.ModuleList([nn.LayerNorm(dim_out) for _ in range(num_layers)])
#
#     def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
#         cov_features = self.cov_layer(tokens)
#         linear_features = self.linear_layer(cov_features)
#         return [self.norm[i](x) for i, x in enumerate(linear_features)]
#
#
# class AdaptedCLIP(nn.Module):
#     def __init__(
#             self,
#             clip_model,
#             text_adapt_weight: float = 0.1,
#             image_adapt_weight: float = 0.1,
#             text_adapt_until: int = 3,
#             image_adapt_until: int = 6,
#             levels: list = [6, 12, 18, 24],
#             relu: bool = True, **kwargs,
#     ):
#         super().__init__()
#         self.clipmodel = clip_model
#         self.image_encoder = clip_model.visual
#         self.text_adapt_until = text_adapt_until
#         self.image_adapt_until = image_adapt_until
#         self.t_w = text_adapt_weight
#         self.i_w = image_adapt_weight
#         self.levels = levels
#         self.num_patch_layers = len(levels)  # 4个patch层
#
#         # 图像适配器
#         layer_adapters = nn.ModuleList(
#             [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
#         )
#
#         # 使用高效多尺度特征提取器
#         self.multi_scale_extractor = MultiScaleFeatureExtractor(1024, 1024, self.num_patch_layers)
#
#         seg_proj = nn.ModuleList(
#             [SimpleProj(1024, 768, relu) for _ in range(self.num_patch_layers)]
#         )
#         det_proj = SimpleProj(1024, 768, relu)
#
#         self.image_adapter = nn.ModuleDict(
#             {
#                 "layer_adapters": layer_adapters,
#                 "seg_proj": seg_proj,
#                 "det_proj": det_proj,
#             }
#         )
#
#         # 文本适配器
#         self.text_adapter = nn.ModuleList(
#             [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
#             + [SimpleProj(768, 768, relu=True)]
#         )
#
#         self._init_weights_()
#
#     def _init_weights_(self):
#         for p in self.image_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for p in self.text_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for p in self.multi_scale_extractor.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward_original(self, x, modality="visual"):
#         if modality == "visual":
#             cls_features, patch_features = self.clipmodel.encode_image(x, [24])
#             patch_features = [
#                 self.clipmodel.visual._global_pool(t)[1] for t in patch_features
#             ]
#             patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
#             patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
#             return patch_features, cls_features
#         else:
#             raise ValueError("modality must be visual")
#
#     def forward(self, x):
#         # 图像编码器前处理
#         x = self.image_encoder.conv1(x)
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         x = x.permute(0, 2, 1)
#
#         x = torch.cat(
#             [
#                 self.image_encoder.class_embedding.to(x.dtype)
#                 + torch.zeros(
#                     x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
#                 ),
#                 x,
#             ],
#             dim=1,
#         )
#         x = x + self.image_encoder.positional_embedding.to(x.dtype)
#
#         x = self.image_encoder.patch_dropout(x)
#         x = self.image_encoder.ln_pre(x)
#         x = x.permute(1, 0, 2)
#
#         # 提取Transformer层特征
#         tokens = []
#         for i in range(24):
#             x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
#             if i < self.image_adapt_until:
#                 adapt_out = self.image_adapter["layer_adapters"][i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.i_w * adapt_out + (1 - self.i_w) * x
#             if i + 1 in self.levels:
#                 tokens.append(x[1:, :, :])  # 收集4个patch层特征 (排除cls token)
#
#         # 处理提取的特征
#         x = x.permute(1, 0, 2)
#         tokens = [t.permute(1, 0, 2) for t in tokens]  # (B, L, C) 其中C=1024
#
#         # 使用高效多尺度特征提取
#         tokens = self.multi_scale_extractor(tokens)
#
#         # 后续处理：归一化和投影
#         tokens = [self.image_encoder.ln_post(t) for t in tokens]
#         seg_tokens = [
#             self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
#         ]
#         seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
#         det_token = self.image_adapter["det_proj"](tokens[-1])
#         det_token = F.normalize(det_token, dim=-1).mean(1)
#         return seg_tokens, det_token
#
#     def encode_text(self, text, adapt_text=True):
#         if not adapt_text:
#             return self.clipmodel.encode_text(text)
#         cast_dtype = self.clipmodel.transformer.get_cast_dtype()
#         x = self.clipmodel.token_embedding(text).to(
#             cast_dtype
#         )  # [batch_size, n_ctx, d_model]
#
#         x = x + self.clipmodel.positional_embedding.to(cast_dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#
#         for i in range(12):
#             x, attn = self.clipmodel.transformer.resblocks[i](
#                 x, attn_mask=self.clipmodel.attn_mask
#             )
#             if i < self.text_adapt_until:
#                 adapt_out = self.text_adapter[i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.t_w * adapt_out + (1 - self.t_w) * x
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#         x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
#         return x



# import torch
# import torch.nn as nn
# import numpy as np
# from torch.nn import functional as F
# from typing import List
# from .adapter_modules import SimpleAdapter, SimpleProj
#
# # 多尺度多形状特征提取模块
# class CovLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, k):
#         super().__init__()
#         self.fc_33 = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=3, padding="same") for _ in range(k)]
#         )
#         self.fc_11 = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=1, padding="same") for _ in range(k)]
#         )
#         self.fc_77 = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=7, padding="same") for _ in range(k)]
#         )
#         self.fc_55 = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=5, padding="same") for _ in range(k)]
#         )
#         self.fc_51 = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=(5, 1), padding="same") for _ in range(k)]
#         )
#         self.fc_15 = nn.ModuleList(
#             [nn.Conv2d(dim_in, dim_out, kernel_size=(1, 5), padding="same") for _ in range(k)]
#         )
#
#     def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
#         processed = []
#         for i, x in enumerate(tokens):
#             if len(x.shape) == 3:  # (B, L, C)
#                 B, L, C = x.shape
#                 H = W = int(np.sqrt(L))
#                 x_reshaped = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
#
#                 # 多尺度卷积融合
#                 conv_out = self.fc_11[i](x_reshaped) + self.fc_33[i](x_reshaped) + \
#                            self.fc_55[i](x_reshaped) + self.fc_77[i](x_reshaped) + \
#                            self.fc_15[i](x_reshaped) + self.fc_51[i](x_reshaped)
#
#                 # 恢复为(B, L, dim_out)
#                 processed_x = conv_out.permute(0, 2, 3, 1).view(B, L, -1)
#                 processed.append(processed_x)
#             else:
#                 raise ValueError(f"Unexpected token shape: {x.shape}")
#         return processed
#
# class LinearLayer(nn.Module):
#     def __init__(self, dim_in, dim_out, k):
#         super().__init__()
#         self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in range(k)])
#
#     def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
#         return [self.fc[i](x) for i, x in enumerate(tokens)]
#
# class MultiScaleFeatureExtractor(nn.Module):
#     def __init__(self, dim_in, dim_out, num_layers):
#         super().__init__()
#         self.cov_layer = CovLayer(dim_in, dim_out, num_layers)
#         self.linear_layer = LinearLayer(dim_out, dim_out, num_layers)
#         self.norm = nn.ModuleList([nn.LayerNorm(dim_out) for _ in range(num_layers)])
#
#     def forward(self, tokens: List[torch.Tensor]) -> List[torch.Tensor]:
#         # 先进行卷积多尺度提取
#         cov_features = self.cov_layer(tokens)
#         # 再进行线性变换
#         linear_features = self.linear_layer(cov_features)
#         # 层归一化
#         return [self.norm[i](x) for i, x in enumerate(linear_features)]
#
# class AdaptedCLIP(nn.Module):
#     def __init__(
#             self,
#             clip_model,
#             text_adapt_weight: float = 0.1,
#             image_adapt_weight: float = 0.1,
#             text_adapt_until: int = 3,
#             image_adapt_until: int = 6,
#             levels: list = [6, 12, 18, 24],
#             relu: bool = True, **kwargs,
#     ):
#         super().__init__()
#         self.clipmodel = clip_model
#         self.image_encoder = clip_model.visual
#         self.text_adapt_until = text_adapt_until
#         self.image_adapt_until = image_adapt_until
#         self.t_w = text_adapt_weight
#         self.i_w = image_adapt_weight
#         self.levels = levels
#         self.num_patch_layers = len(levels)  # 4个patch层
#
#         # 图像适配器
#         layer_adapters = nn.ModuleList(
#             [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
#         )
#
#         # 初始化多尺度多形状特征提取模块 (输入1024维，输出1024维，4个patch层)
#         self.multi_scale_extractor = MultiScaleFeatureExtractor(1024, 1024, self.num_patch_layers)
#
#         seg_proj = nn.ModuleList(
#             [SimpleProj(1024, 768, relu) for _ in range(self.num_patch_layers)]
#         )
#         det_proj = SimpleProj(1024, 768, relu)
#
#         self.image_adapter = nn.ModuleDict(
#             {
#                 "layer_adapters": layer_adapters,
#                 "seg_proj": seg_proj,
#                 "det_proj": det_proj,
#             }
#         )
#
#         # 文本适配器
#         self.text_adapter = nn.ModuleList(
#             [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
#             + [SimpleProj(768, 768, relu=True)]
#         )
#
#         self._init_weights_()
#
#     def _init_weights_(self):
#         for p in self.image_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for p in self.text_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         # 初始化多尺度模块权重
#         for p in self.multi_scale_extractor.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward_original(self, x, modality="visual"):
#         if modality == "visual":
#             cls_features, patch_features = self.clipmodel.encode_image(x, [24])
#             patch_features = [
#                 self.clipmodel.visual._global_pool(t)[1] for t in patch_features
#             ]
#             patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
#             patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
#             return patch_features, cls_features
#         else:
#             raise ValueError("modality must be visual")
#
#     def forward(self, x):
#         # 图像编码器前处理
#         x = self.image_encoder.conv1(x)
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         x = x.permute(0, 2, 1)
#
#         x = torch.cat(
#             [
#                 self.image_encoder.class_embedding.to(x.dtype)
#                 + torch.zeros(
#                     x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
#                 ),
#                 x,
#             ],
#             dim=1,
#         )
#         x = x + self.image_encoder.positional_embedding.to(x.dtype)
#
#         x = self.image_encoder.patch_dropout(x)
#         x = self.image_encoder.ln_pre(x)
#         x = x.permute(1, 0, 2)
#
#         # 提取Transformer层特征
#         tokens = []
#         for i in range(24):
#             x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
#             if i < self.image_adapt_until:
#                 adapt_out = self.image_adapter["layer_adapters"][i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.i_w * adapt_out + (1 - self.i_w) * x
#             if i + 1 in self.levels:
#                 tokens.append(x[1:, :, :])  # 收集4个patch层特征 (排除cls token)
#
#         # 处理提取的特征
#         x = x.permute(1, 0, 2)
#         tokens = [t.permute(1, 0, 2) for t in tokens]  # (B, L, C) 其中C=1024
#
#         # 插入多尺度多形状特征提取 (在4个patch层之后)
#         tokens = self.multi_scale_extractor(tokens)
#
#         # 后续处理：归一化和投影
#         tokens = [self.image_encoder.ln_post(t) for t in tokens]
#         seg_tokens = [
#             self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
#         ]
#         seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
#         det_token = self.image_adapter["det_proj"](tokens[-1])
#         det_token = F.normalize(det_token, dim=-1).mean(1)
#         return seg_tokens, det_token
#
#     def encode_text(self, text, adapt_text=True):
#         if not adapt_text:
#             return self.clipmodel.encode_text(text)
#         cast_dtype = self.clipmodel.transformer.get_cast_dtype()
#         x = self.clipmodel.token_embedding(text).to(
#             cast_dtype
#         )  # [batch_size, n_ctx, d_model]
#
#         x = x + self.clipmodel.positional_embedding.to(cast_dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#
#         for i in range(12):
#             x, attn = self.clipmodel.transformer.resblocks[i](
#                 x, attn_mask=self.clipmodel.attn_mask
#             )
#             if i < self.text_adapt_until:
#                 adapt_out = self.text_adapter[i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.t_w * adapt_out + (1 - self.t_w) * x
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#         x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
#         return x



# import torch
# from torch import nn
# import torch.nn.functional as F
# from .adapter_modules import SimpleAdapter, SimpleProj
#
#
# # 多尺度特征投影模块
# class MultiScaleProjection(nn.Module):
#     """多尺度特征投影模块（修正因果卷积）"""
#
#     def __init__(self, d_model, scales=[3, 5, 7], dilation_rates=[1, 2, 3]):
#         super().__init__()
#         self.scales = scales
#         self.dilation_rates = dilation_rates
#         self.projections = nn.ModuleList()
#         for scale, dilation in zip(scales, dilation_rates):
#             conv = nn.Conv1d(
#                 in_channels=d_model,
#                 out_channels=d_model * 3,  # 同时生成Q,K,V
#                 kernel_size=scale,
#                 padding=0,  # 手动左侧填充以实现因果卷积
#                 dilation=dilation
#             )
#             self.projections.append(conv)
#
#     def forward(self, x):
#         """输入x: [B, L, D]，输出: 多尺度Q,K,V列表"""
#         x = x.transpose(1, 2)  # [B, D, L]
#         projections = []
#         for proj, (scale, dilation) in zip(self.projections, zip(self.scales, self.dilation_rates)):
#             pad_amount = (scale - 1) * dilation
#             x_padded = F.pad(x, (pad_amount, 0))  # 左侧填充
#             out = proj(x_padded)  # [B, 3*D, L]
#             out = out.transpose(1, 2)  # [B, L, 3*D]
#             q, k, v = torch.chunk(out, 3, dim=-1)
#             projections.append((q, k, v))
#         return projections
#
#
# # 多尺度自适应注意力头
# class MultiScaleAdaptiveAttentionHead(nn.Module):
#     """多尺度自适应注意力头"""
#
#     def __init__(self, head_dim, control_dim, n_scales=3):
#         super().__init__()
#         self.head_dim = head_dim
#         self.n_scales = n_scales
#
#         # 多尺度投影网络
#         self.multi_scale_proj = MultiScaleProjection(head_dim)
#
#         # 动态门控网络
#         self.gate_network = nn.Sequential(
#             nn.Linear(control_dim, 4 * control_dim),
#             nn.GELU(),
#             nn.Linear(4 * control_dim, n_scales),
#             nn.Softmax(dim=-1)
#         )
#
#     def forward(self, x, control_vector):
#         """
#         输入:
#             x: [batch_size, seq_len, head_dim]
#             control_vector: [batch_size, control_dim]
#         输出:
#             加权融合后的注意力结果 [batch_size, seq_len, head_dim]
#         """
#         # 生成多尺度Q,K,V
#         scale_projections = self.multi_scale_proj(x)
#
#         # 计算各尺度的注意力结果
#         attn_outputs = []
#         for q, k, v in scale_projections:
#             scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
#             attn_weights = F.softmax(scores, dim=-1)
#             attn = torch.matmul(attn_weights, v)
#             attn_outputs.append(attn)
#
#         # 动态门控权重
#         gate_weights = self.gate_network(control_vector)  # [B, n_scales]
#
#         # 加权融合
#         combined_output = torch.zeros_like(attn_outputs[0])
#         for i in range(self.n_scales):
#             combined_output += gate_weights[:, i].unsqueeze(1).unsqueeze(2) * attn_outputs[i]
#
#         return combined_output
#
#
# # 完整的多头自适应注意力机制
# class AdaptiveMultiHeadAttention(nn.Module):
#     """完整的多头自适应注意力机制"""
#
#     def __init__(self, d_model=512, n_heads=8, n_scales=3):
#         super().__init__()
#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.head_dim = d_model // n_heads
#
#         # 多个并行注意力头
#         self.heads = nn.ModuleList([
#             MultiScaleAdaptiveAttentionHead(self.head_dim, d_model, n_scales)
#             for _ in range(n_heads)
#         ])
#
#         # 控制向量生成网络
#         self.control_net = nn.LSTM(
#             input_size=d_model,
#             hidden_size=d_model,
#             num_layers=1,
#             bidirectional=True
#         )
#
#         # 输出投影
#         self.out_proj = nn.Linear(d_model, d_model)
#
#     def forward(self, x):
#         batch_size, seq_len, _ = x.shape
#
#         # 生成控制向量
#         _, (h_n, _) = self.control_net(x.transpose(0, 1))
#         control_vector = h_n.mean(dim=0)  # [B, D]
#
#         # 分割为多头
#         x = x.view(batch_size, seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
#
#         # 各头并行计算
#         head_outputs = []
#         for i, head in enumerate(self.heads):
#             head_x = x[:, i, :, :]  # [B, L, head_dim]
#             head_out = head(head_x, control_vector)
#             head_outputs.append(head_out)
#
#         # 合并多头结果
#         combined = torch.cat(head_outputs, dim=-1)  # [B, L, D]
#         output = self.out_proj(combined)
#
#         return output
#
#
# class AdaptedCLIP(nn.Module):
#     def __init__(
#             self,
#             clip_model,
#             text_adapt_weight: float = 0.1,
#             image_adapt_weight: float = 0.1,
#             text_adapt_until: int = 3,
#             image_adapt_until: int = 6,
#             levels: list = [6, 12, 18, 24],
#             relu: bool = True,
#             # 新增参数：多尺度注意力配置
#             attention_d_model: int = 1024,
#             attention_n_heads: int = 8,
#             attention_n_scales: int = 3,
#             **kwargs,
#     ):
#         super().__init__()
#         self.clipmodel = clip_model
#         self.image_encoder = clip_model.visual
#         self.text_adapt_until = text_adapt_until
#         self.image_adapt_until = image_adapt_until
#         self.t_w = text_adapt_weight
#         self.i_w = image_adapt_weight
#         self.levels = levels
#
#         # 初始化多尺度多头自适应注意力模块
#         self.multi_scale_attention = nn.ModuleList([
#             AdaptiveMultiHeadAttention(
#                 d_model=attention_d_model,
#                 n_heads=attention_n_heads,
#                 n_scales=attention_n_scales
#             ) for _ in range(len(levels))  # 为每个层级创建一个注意力模块
#         ])
#
#         layer_adapters = nn.ModuleList(
#             [SimpleAdapter(1024, 1024) for _ in range(image_adapt_until)]
#         )
#         seg_proj = nn.ModuleList(
#             [SimpleProj(1024, 768, relu) for _ in range(len(levels))]
#         )
#         det_proj = SimpleProj(1024, 768, relu)
#         self.image_adapter = nn.ModuleDict(
#             {
#                 "layer_adapters": layer_adapters,
#                 "seg_proj": seg_proj,
#                 "det_proj": det_proj,
#             }
#         )
#         self.text_adapter = nn.ModuleList(
#             [SimpleAdapter(768, 768) for _ in range(text_adapt_until)]
#             + [SimpleProj(768, 768, relu=True)]
#         )
#         self._init_weights_()
#
#     def _init_weights_(self):
#         for p in self.image_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         for p in self.text_adapter.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#         # 初始化注意力模块参数
#         for p in self.multi_scale_attention.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)
#
#     def forward_original(self, x, modality="visual"):
#         if modality == "visual":
#             cls_features, patch_features = self.clipmodel.encode_image(x, [24])
#             patch_features = [
#                 self.clipmodel.visual._global_pool(t)[1] for t in patch_features
#             ]
#             patch_features = [self.clipmodel.visual.ln_post(t) for t in patch_features]
#             patch_features = [t @ self.clipmodel.visual.proj for t in patch_features]
#             return patch_features, cls_features
#         else:
#             raise ValueError("modality must be visual")
#
#     def forward(self, x):
#         x = self.image_encoder.conv1(x)
#         x = x.reshape(x.shape[0], x.shape[1], -1)
#         x = x.permute(0, 2, 1)
#
#         x = torch.cat(
#             [
#                 self.image_encoder.class_embedding.to(x.dtype)
#                 + torch.zeros(
#                     x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
#                 ),
#                 x,
#             ],
#             dim=1,
#         )
#         x = x + self.image_encoder.positional_embedding.to(x.dtype)
#
#         x = self.image_encoder.patch_dropout(x)
#         x = self.image_encoder.ln_pre(x)
#
#         x = x.permute(1, 0, 2)
#
#         tokens = []
#         for i in range(24):
#             x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
#             if i < self.image_adapt_until:
#                 adapt_out = self.image_adapter["layer_adapters"][i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.i_w * adapt_out + (1 - self.i_w) * x
#             if i + 1 in self.levels:
#                 tokens.append(x[1:, :, :])  # 获取patch特征（排除cls token）
#
#         # 将收集到的tokens进行处理
#         x = x.permute(1, 0, 2)
#
#         # 对每个层级的tokens应用多尺度多头自适应注意力（MMCI思想）
#         processed_tokens = []
#         for i, t in enumerate(tokens):
#             # 调整维度 [L, B, D] -> [B, L, D]
#             token = t.permute(1, 0, 2)
#
#             # 应用多尺度注意力模块（核心修改点）
#             attn_token = self.multi_scale_attention[i](token)
#
#             # 残差连接
#             attn_token = attn_token + token
#
#             processed_tokens.append(attn_token)
#
#         # 继续原有的处理流程
#         tokens = [self.image_encoder.ln_post(t) for t in processed_tokens]
#         seg_tokens = [
#             self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
#         ]
#         seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
#         det_token = self.image_adapter["det_proj"](tokens[-1])
#         det_token = F.normalize(det_token, dim=-1).mean(1)
#         return seg_tokens, det_token
#
#     def encode_text(self, text, adapt_text=True):
#         if not adapt_text:
#             return self.clipmodel.encode_text(text)
#         cast_dtype = self.clipmodel.transformer.get_cast_dtype()
#         x = self.clipmodel.token_embedding(text).to(
#             cast_dtype
#         )  # [batch_size, n_ctx, d_model]
#
#         x = x + self.clipmodel.positional_embedding.to(cast_dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#
#         for i in range(12):
#             x, attn = self.clipmodel.transformer.resblocks[i](
#                 x, attn_mask=self.clipmodel.attn_mask
#             )
#             if i < self.text_adapt_until:
#                 adapt_out = self.text_adapter[i](x)
#                 adapt_out = (
#                         adapt_out
#                         * x.norm(dim=-1, keepdim=True)
#                         / adapt_out.norm(dim=-1, keepdim=True)
#                 )
#                 x = self.t_w * adapt_out + (1 - self.t_w) * x
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.clipmodel.ln_final(x)  # [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding
#         x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
#         return x