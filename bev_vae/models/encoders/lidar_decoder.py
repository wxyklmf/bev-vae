import torch
import torch.nn as nn
import numpy as np
from bev_vae.models.encoders.swin_layers import BasicLayer

# ============================================================
# 原始代码来自 UltraLiDAR/plugin/models/necks/vq_layer.py
# 中的 VQDecoder 类
#
# 删除：@MODELS.register_module()
# 删除：from mmdet3d.registry import MODELS
# 原因：不需要 mmdet3d 注册机制
# ============================================================


class VQDecoder(nn.Module):
    """
    将量化特征解码回体素网格。
    
    输入: (B, L, codebook_dim)  例如 (B, 6400, 1024)
    输出: (B, z_size, y_size, x_size)  例如 (B, 40, 640, 640)
    """
    def __init__(
        self,
        img_size=640,
        num_patches=6400,
        patch_size=8,
        in_chans=40,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
        bias_init=-3,
    ):
        super().__init__()

        norm_layer = nn.LayerNorm
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.h = img_size // patch_size   # 80
        self.w = img_size // patch_size   # 80
        self.num_patches = num_patches

        # 将码本维度映射回 Transformer 隐藏维度
        self.decoder_embed = nn.Linear(codebook_dim, embed_dim, bias=True)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )

        # Swin Transformer 块
        self.blocks = BasicLayer(
            dim=embed_dim,
            input_resolution=(self.h, self.w),
            depth=depth,
            num_heads=num_heads,
            window_size=8,
        )

        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        
        # 预测每个 patch 的体素值
        self.pred = nn.Linear(embed_dim, patch_size**2 * in_chans, bias=True)
        
        self._initialize_weights()
        nn.init.constant_(self.pred.bias, bias_init)

    def _initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.h, self.w), cls_token=False
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        将 patch 序列重新拼接为体素网格。
        
        输入: (B, L, patch_size^2 * in_chans)
        输出: (B, in_chans, H, W)
        """
        p = self.patch_size
        h, w = self.h, self.w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))

        return imgs

    def forward(self, x):
        """
        forward 逻辑：
        1. 嵌入到 embed_dim
        2. 加位置编码
        3. Swin Transformer
        4. LayerNorm
        5. 预测每个 patch 的像素值
        6. 重新拼接为网格
        """
        x = self.decoder_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pred(x)
        x = self.unpatchify(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
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
