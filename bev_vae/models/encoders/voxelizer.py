import torch
import torch.nn as nn
import numpy as np


class Voxelizer(nn.Module):
    """
    将点云转换为 BEV 体素网格。
    
    输入：点云列表 List[Tensor[N, 3]]，每个 Tensor 是一个样本的点云
    输出：BEV Voxel Tensor (B, z_size, y_size, x_size)
    
    参数说明：
        x_min, x_max, y_min, y_max: BEV 平面范围（米）
        z_min, z_max: 高度范围（米）
        step: 水平体素大小（米），UltraLiDAR 默认 0.15625
        z_step: 垂直体素大小（米），UltraLiDAR 默认 0.2
    """
    def __init__(
        self,
        x_min=-50.0,
        x_max=50.0,
        y_min=-50.0,
        y_max=50.0,
        z_min=-3.0,
        z_max=5.0,
        step=0.15625,
        z_step=0.2,
    ):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max
        self.step = step
        self.z_step = z_step

        # 计算网格尺寸
        self.x_size = int(np.ceil((x_max - x_min) / step))
        self.y_size = int(np.ceil((y_max - y_min) / step))
        self.z_size = int(np.ceil((z_max - z_min) / z_step))

    def forward(self, points):
        """
        将点云列表转换为体素网格。
        
        Args:
            points: List[torch.Tensor]，每个 Tensor 形状为 [N_i, 3] 或 [N_i, 4]
                    取前 3 列 (x, y, z)
        
        Returns:
            torch.Tensor: (B, z_size, y_size, x_size)，占据体素为 1，否则为 0
        """
        batch_size = len(points)
        # 初始化全零体素网格
        voxel_grid = torch.zeros(
            (batch_size, self.z_size, self.y_size, self.x_size),
            dtype=torch.float32,
            device=points[0].device,
        )

        for i, pts in enumerate(points):
            # 取 x, y, z
            pts = pts[:, :3].cpu().numpy()

            # 计算每个点在体素网格中的索引
            x_idx = ((pts[:, 0] - self.x_min) / self.step).astype(np.int32)
            y_idx = ((pts[:, 1] - self.y_min) / self.step).astype(np.int32)
            z_idx = ((pts[:, 2] - self.z_min) / self.z_step).astype(np.int32)

            # 过滤超出网格范围的点
            valid = (
                (x_idx >= 0) & (x_idx < self.x_size)
                & (y_idx >= 0) & (y_idx < self.y_size)
                & (z_idx >= 0) & (z_idx < self.z_size)
            )
            x_idx = x_idx[valid]
            y_idx = y_idx[valid]
            z_idx = z_idx[valid]

            # 标记占据的体素
            voxel_grid[i, z_idx, y_idx, x_idx] = 1.0

        return voxel_grid
