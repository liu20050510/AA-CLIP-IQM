import math

import torch
from torch import nn
import torch.nn.functional as F
from .adapter_modules import SimpleAdapter, SimpleProj
from .iqm import IQM, IQMConfig  # 导入IQ模块


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
            # IQM相关参数
            iqm_hidden_size: int = 768,
            iqm_num_layers: int = 4,
            iqm_num_heads: int = 8, **kwargs,
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

        # 初始化IQ模块
        self.iqm_config = IQMConfig(
            hidden_size=iqm_hidden_size,
            num_hidden_layers=iqm_num_layers,
            num_attention_heads=iqm_num_heads,
            encoder_hidden_size=iqm_hidden_size,  # 图像特征维度
            text_encoder_hidden_size=768,  # 文本特征维度
        )
        self.iqm = IQM(self.iqm_config)

        # 类别查询MLP
        self.class_query_mlp = nn.Sequential(
            nn.Linear(1024, iqm_hidden_size),
            nn.ReLU(),
            nn.Linear(iqm_hidden_size, iqm_hidden_size)
        )

        # 查询适配器 - 用于将视觉特征投影到查询空间
        self.query_adapters = nn.ModuleList(
            [SimpleProj(1024, iqm_hidden_size, relu) for _ in range(len(levels))]
        )

        # 视觉特征投影器参数
        self.iqm_hidden_size = iqm_hidden_size
        self.visual_feature_proj = None

        # 文本特征投影器
        self.text_feature_proj = None

        # 位置嵌入
        self.pos_embedding = self._create_positional_embedding(max_len=512, d_model=iqm_hidden_size)

        # 添加可学习的融合权重参数，并增加约束
        self.visual_weight = nn.Parameter(torch.tensor(0.5))
        self.text_weight = nn.Parameter(torch.tensor(0.5))

        # 添加IQM输出的正则化参数
        self.iqm_dropout = nn.Dropout(0.1)
        self.iqm_layer_norm = nn.LayerNorm(iqm_hidden_size)

        self._init_weights_()

    def _create_positional_embedding(self, max_len, d_model):
        """创建正弦位置嵌入"""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe.permute(1, 0, 2))  # 形状: [1, max_len, d_model]

    def _init_weights_(self):
        for p in self.image_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.text_adapter.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # 初始化IQM参数
        for p in self.iqm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.class_query_mlp.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.query_adapters.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    def forward(self, x, text_embeddings=None, iqm_hidden_size=None):
        # 提取图像特征
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
                tokens.append(x[1:, :, :])  # 保存除CLS外的特征

        x = x.permute(1, 0, 2)
        tokens = [t.permute(1, 0, 2) for t in tokens]
        tokens = [self.image_encoder.ln_post(t) for t in tokens]

        # 准备用于分割和检测的特征
        seg_tokens = [
            self.image_adapter["seg_proj"][i](t) for i, t in enumerate(tokens)
        ]
        seg_tokens = [F.normalize(t, dim=-1) for t in seg_tokens]
        det_token = self.image_adapter["det_proj"](tokens[-1])
        det_token = F.normalize(det_token, dim=-1).mean(1)

        # IQM处理流程
        iqm_outputs = None
        if text_embeddings is not None:
            # 1. 位置-类别查询初始化
            # 获取CLS特征作为类别相关视觉特征
            cls_feature = x[:, 0, :]  # [batch_size, hidden_size]
            class_query = self.class_query_mlp(cls_feature).unsqueeze(1)  # [batch_size, 1, iqm_hidden_size]

            # 为正常和异常创建两个查询
            # 重复查询以创建两个查询嵌入：一个用于正常，一个用于异常
            class_query = class_query.repeat(1, 2, 1)  # [batch_size, 2, iqm_hidden_size]

            # 注入位置信息
            batch_size = class_query.shape[0]
            seq_len = class_query.shape[1]
            pos_emb = self.pos_embedding[:, :seq_len, :].repeat(batch_size, 1,
                                                                1)  # [batch_size, seq_len, iqm_hidden_size]
            query_embeds = class_query + pos_emb  # 初始查询嵌入F_0^Q

            # 2. 视觉特征预处理：投影到查询空间
            projected_visual_features = [
                self.query_adapters[i](t) for i, t in enumerate(tokens)
            ]
            # 拼接多阶段特征
            concatenated_visual = torch.cat(projected_visual_features,
                                            dim=1)  # [batch_size, total_patches, iqm_hidden_size * num_levels]

            # 动态创建视觉特征投影器
            in_features = int(concatenated_visual.shape[-1])
            if self.visual_feature_proj is None or self.visual_feature_proj.in_features != in_features:
                # 使用一个简单的函数来创建线性层，避免参数类型问题
                self.visual_feature_proj = nn.Linear(in_features, self.iqm_hidden_size)
                self.visual_feature_proj = self.visual_feature_proj.to(concatenated_visual.device)

            # 投影到 IQM 期望的维度
            projected_concatenated_visual = self.visual_feature_proj(concatenated_visual)

            # 3. 多模态特征交互
            # 调整text_embeddings的维度以匹配IQM期望的输入格式
            # 确保 text_embeddings 有正确的维度 [batch_size, seq_len, hidden_size]
            if text_embeddings.dim() == 2:
                # 如果是 [hidden_size, 2] 形状，转置并扩展为 [batch_size, 2, hidden_size]
                adjusted_text_embeddings = text_embeddings.transpose(0, 1).unsqueeze(0).repeat(x.size(0), 1, 1)
            elif text_embeddings.dim() == 3:
                if text_embeddings.shape[1] == 2:
                    # 如果是 [batch_size, 2, hidden_size] 形状，保持不变
                    adjusted_text_embeddings = text_embeddings
                else:
                    # 其他情况，可能需要调整
                    adjusted_text_embeddings = text_embeddings
            else:
                # 其他情况，添加一个维度并扩展到正确的batch_size
                adjusted_text_embeddings = text_embeddings.unsqueeze(0).repeat(x.size(0), 1, 1)

            # 确保文本特征维度与 IQM 配置匹配
            text_hidden_size = adjusted_text_embeddings.shape[-1]
            if self.text_feature_proj is None or self.text_feature_proj.in_features != text_hidden_size:
                self.text_feature_proj = nn.Linear(text_hidden_size, 768).to(adjusted_text_embeddings.device)

            # 投影文本特征到 IQM 期望的维度
            projected_text_embeddings = self.text_feature_proj(adjusted_text_embeddings)

            # 应用可学习的自适应加权融合
            visual_weight_normalized = torch.sigmoid(self.visual_weight)
            text_weight_normalized = torch.sigmoid(self.text_weight)

            # 归一化权重，使它们的和为1
            weight_sum = visual_weight_normalized + text_weight_normalized
            visual_weight_normalized = visual_weight_normalized / weight_sum
            text_weight_normalized = text_weight_normalized / weight_sum

            iqm_outputs = self.iqm(
                query_embeds=query_embeds,
                query_length=query_embeds.shape[1],
                encoder_hidden_states=projected_concatenated_visual * visual_weight_normalized,
                text_encoder_hidden_states=projected_text_embeddings * text_weight_normalized,
            )

            # 对IQM输出进行正则化
            iqm_outputs.last_hidden_state = self.iqm_layer_norm(iqm_outputs.last_hidden_state)
            iqm_outputs.last_hidden_state = self.iqm_dropout(iqm_outputs.last_hidden_state)

            # 获取最终查询嵌入
            final_query_embedding = iqm_outputs.last_hidden_state  # [batch_size, query_len, iqm_hidden_size]

        return seg_tokens, det_token, iqm_outputs

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
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = self.text_adapter[-1](x[torch.arange(x.shape[0]), text.argmax(dim=-1)])
        # x = (
        # x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        # @ self.clipmodel.text_projection
        # )
        return x
