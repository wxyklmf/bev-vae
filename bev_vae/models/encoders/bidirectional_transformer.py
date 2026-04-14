import torch
import torch.nn as nn
import numpy as np
from bev_vae.models.encoders.swin_layers import BasicLayer

# ============================================================
# 原始代码来自 UltraLiDAR/plugin/models/necks/vq_layer.py
# 中的 BidirectionalTransformer 类
#
# 删除：@MODELS.register_module()
# 删除：from mmdet3d.registry import MODELS
# 原因：不需要 mmdet3d 注册机制
# ============================================================


class BidirectionalTransformer(nn.Module):
    """
    双向 Transformer，用于 Stage2 的码本预测（MaskGIT 风格）。
    
    输入: 部分掩码的码本特征 (B, L, e_dim)
    输出: 每个位置的码本类别预测 (B, L, n_e)
    
    用途：
    训练时：遮住一部分码本索引，让模型预测被遮住的部分
    生成时：从全掩码开始，逐步预测并解码出完整码本
    """
    def __init__(
        self,
        n_e=1024,             # 码本大小（预测的类别数）
        e_dim=1024,           # 输入特征维度
        img_size=80,          # BEV 网格尺寸（patch 数量 = img_size^2）
        hidden_dim=512,       # Transformer 隐藏维度
        depth=24,             # Transformer 层数
        num_heads=16,         # 注意力头数
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.img_size = img_size
        self.hidden_dim = hidden_dim
        
        # ============================================================
        # 以下所有代码与原始 UltraLiDAR 完全一致
        # 只删除了 @MODELS.register_module() 装饰器
        # ============================================================
        
        # 将码本特征投影到 Transformer 隐藏维度
        self.decoder_embed = nn.Linear(e_dim, hidden_dim, bias=True)
        
        # mask token：用于替换被��住位置的占位符
        self.mask_token = nn.Parameter(torch.zeros(1, 1, e_dim), requires_grad=True)
        
        token_size = img_size**2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, token_size, hidden_dim), requires_grad=False
        )
        
        # Swin Transformer 块
        self.blocks = BasicLayer(
            dim=hidden_dim,
            input_resolution=(img_size, img_size),
            depth=depth,
            num_heads=num_heads,
            window_size=8,
            downsample=None,
        )
        
        self.norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.GELU())
        self.pred = nn.Linear(hidden_dim, n_e, bias=True)
        
        self.initialize_weights()

    def initialize_weights(self):
        # 用 2D 正弦位置编码初始化 pos_embed
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_dim, (self.img_size, self.img_size), cls_token=False
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float())

        # 初始化 mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Args:
            x: 部分掩码的码本特征 (B, L, e_dim)
               其中被遮住的位置用 mask_token 填充
        
        Returns:
            预测的码本类别 logits (B, L, n_e)
        """
        x = self.decoder_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pred(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """与之前完全一致"""
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb
