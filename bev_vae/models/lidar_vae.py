import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from bev_vae.models.encoders.voxelizer import Voxelizer
from bev_vae.models.encoders.lidar_encoder import VQEncoder
from bev_vae.models.encoders.lidar_decoder import VQDecoder
from bev_vae.models.encoders.vq_layer import VectorQuantizer


class LiDARVAE(pl.LightningModule):
    """
    点云 VQ-VAE 模型，集成到 PyTorch Lightning 框架。
    
    数据流：
    点云 → Voxelizer → VQEncoder → VectorQuantizer → VQDecoder → 体素重建
    
    训练目标：让重建的体素尽可能接近原始体素
    """
    
    def __init__(
        self,
        # Voxelizer 参数
        x_min=-50.0,
        x_max=50.0,
        y_min=-50.0,
        y_max=50.0,
        z_min=-3.0,
        z_max=5.0,
        voxel_step=0.15625,
        z_step=0.2,
        # Encoder/Decoder 参数
        img_size=640,
        patch_size=8,
        embed_dim=512,
        num_heads=16,
        encoder_depth=12,
        decoder_depth=12,
        # VQ 参数
        codebook_dim=1024,
        n_e=1024,
        beta=1.0,
        # 训练参数
        learning_rate=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()  # 保存所有超参数到 checkpoint
        
        # ============================================================
        # 第 1 步：Voxelizer（点云 → 体素网格）
        # ============================================================
        self.voxelizer = Voxelizer(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_min=z_min, z_max=z_max,
            step=voxel_step,
            z_step=z_step,
        )
        
        # 计算 z 维度大小（= 体素网格的深度通道数）
        z_size = self.voxelizer.z_size
        
        # ============================================================
        # 第 2 步：VQEncoder（体素 → 特征）
        # ============================================================
        self.encoder = VQEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=z_size,        # 输入通道数 = z 维度的体素数
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=encoder_depth,
            codebook_dim=codebook_dim,
        )
        
        # ============================================================
        # 第 3 步：VectorQuantizer（特征 → 量化码本）
        # ============================================================
        self.vq = VectorQuantizer(
            n_e=n_e,
            e_dim=codebook_dim,
            beta=beta,
        )
        
        # ============================================================
        # 第 4 步：VQDecoder（量化特征 → 体素重建）
        # ============================================================
        self.decoder = VQDecoder(
            img_size=img_size,
            num_patches=(img_size // patch_size) ** 2,
            patch_size=patch_size,
            in_chans=z_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            depth=decoder_depth,
            codebook_dim=codebook_dim,
        )
        
        self.learning_rate = learning_rate

    def forward(self, points):
        """
        前向传播：点云 → 体素 → 特征 → 量化 → 重建体素
        
        Args:
            points: List[torch.Tensor]，每个 Tensor 形状为 [N_i, 3]
        
        Returns:
            recon_voxels: 重建的体素网格 (B, z_size, y_size, x_size)
            vq_loss: 量化损失
            code_indices: 码本索引 (B, L)
        """
        # 第 1 步：点云 → 体素
        voxels = self.voxelizer(points)
        
        # 第 2 步：体素 → 特征
        feats = self.encoder(voxels)
        
        # 第 3 步：特征 → 量化
        quant, vq_loss_tuple, indices = self.vq(feats)
        
        # 第 4 步：量化特征 → 重建体素
        recon_voxels = self.decoder(quant)
        
        # 量化损失 = commitment loss + codebook loss
        vq_loss = sum(vq_loss_tuple)
        
        return recon_voxels, vq_loss, indices

    def training_step(self, batch, batch_idx):
        """
        训练一步：计算 loss 并返回
        
        Args:
            batch: 包含 "points" 的字典
            batch_idx: 当前 batch 索引（Lightning 自动传入）
        
        Returns:
            loss: 用于反向传播的总 loss
        """
        points = batch["points"]
        
        # 前向传播
        recon_voxels, vq_loss, indices = self(points)
        
        # 重建 loss：BCE（因为体素是 0/1 的二值网格）
        # 注意：voxelizer 输出的是 0/1，所以用 BCE
        recon_loss = F.binary_cross_entropy_with_logits(
            recon_voxels, 
            self.voxelizer(points),  # 这里会再算一次 voxelizer，可以优化
        )
        
        # 总 loss = 重建 loss + 量化 loss
        loss = recon_loss + vq_loss
        
        # 记录 loss（Lightning 自动处理 tensorboard/wandb 日志）
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/recon_loss", recon_loss)
        self.log("train/vq_loss", vq_loss)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """验证一步"""
        points = batch["points"]
        recon_voxels, vq_loss, indices = self(points)
        
        recon_loss = F.binary_cross_entropy_with_logits(
            recon_voxels,
            self.voxelizer(points),
        )
        
        loss = recon_loss + vq_loss
        
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/recon_loss", recon_loss)
        self.log("val/vq_loss", vq_loss)
        
        return loss

    def configure_optimizers(self):
        """
        配置优化器。
        Lightning 调用这个方法来获取优化器。
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
