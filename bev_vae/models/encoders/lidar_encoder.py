import torch
import torch.nn as nn
import numpy as np
from bev_vae.models.encoders.swin_layers import BasicLayer
from timm.models.vision_transformer import PatchEmbed

# ============================================================
# 原始代码（来自 UltraLiDAR/plugin/models/necks/vq_layer.py）：
# 
# from mmdet3d.registry import MODELS                     <- 删除
# from mmcv.cnn.bricks.transformer import MultiheadAttention  <- 删除
# from mmcv.cnn.bricks.transformer import FFN                 <- 删除
#
# 原因：这些是 mmdet3d/mmcv 的依赖，我们改用纯 PyTorch
# ============================================================


# 原始代码：@MODELS.register_module()    <- 删除了这个装饰器
# 原因：@register_module() 是 mmdet3d ���注册机制，
#       让框架能通过配置字典自动构建模型。
#       我们手动实例化，不需要注册。
class VQEncoder(nn.Module):
    """
    将体素网格编码为特征向量。
    
    输入: (B, z_size, y_size, x_size)  例如 (B, 40, 640, 640)
    输出: (B, L, codebook_dim)         例如 (B, 6400, 1024)
    """
    def __init__(
        self,
        img_size=640,
        patch_size=8,
        in_chans=40,
        embed_dim=512,
        num_heads=16,
        depth=12,
        codebook_dim=1024,
    ):
        super().__init__()
        
        norm_layer = nn.LayerNorm
        
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        num_patches = self.patch_embed.num_patches
        
        self.h = img_size // patch_size
        self.w = img_size // patch_size
        
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )
        
        # 原始代码：
        # self.blocks = [
        #     BasicLayer(
        #         embed_dim,
        #         (img_size // patch_size, img_size // patch_size),
        #         depth,
        #         num_heads=num_heads,
        #         window_size=8,
        #         downsample=None,
        #     ),
        # ]
        # self.blocks = nn.Sequential(*self.blocks)
        #
        # 修改：去掉了列表包裹和解包，直接赋值
        self.blocks = BasicLayer(
            dim=embed_dim,
            input_resolution=(img_size // patch_size, img_size // patch_size),
            depth=depth,
            num_heads=num_heads,
            window_size=8,
            downsample=None,
        )
        
        self.norm = nn.Sequential(norm_layer(embed_dim), nn.GELU())
        self.pre_quant = nn.Linear(embed_dim, codebook_dim)
        
        self._initialize_weights()

    def _initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], (self.h, self.w), cls_token=False
        )
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )
        
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
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
        # forward 逻辑与原始代码完全一致
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_quant(x)
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
