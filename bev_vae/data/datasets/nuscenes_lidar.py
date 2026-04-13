from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class nuScenesLidarDataset(Dataset):
    """
    读取 nuScenes 原始点云 .bin 文件的 Dataset。
    每个 .bin 文件包含 N 行 4 列：x, y, z, intensity（float32）。
    返回的点云格式：torch.Tensor[N, 3]（只取 x, y, z）
    这与 UltraLiDAR 的 forward_train(points) 接口一致。
    """
    data_dir: Path
    file_list: List[str]
    max_range: float = 50.0
    min_range: float = 1.0

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.valid_files = []
        for f in self.file_list:
            full_path = self.data_dir / f
            if full_path.exists():
                self.valid_files.append(f)
        if len(self.valid_files) == 0:
            raise FileNotFoundError(
                f"No .bin files found in {self.data_dir}. "
                f"Please check if the data directory is correct."
            )

    def __len__(self) -> int:
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Dict:
        filename = self.valid_files[idx]
        full_path = self.data_dir / filename
        points = np.fromfile(str(full_path), dtype=np.float32).reshape(-1, 4)
        xyz = points[:, :3]
        dists = np.linalg.norm(xyz, axis=1)
        mask = (dists >= self.min_range) & (dists <= self.max_range)
        xyz = xyz[mask]
        points_tensor = torch.from_numpy(xyz).float()
        return {
            "points": points_tensor,
            "filename": filename,
            "num_points": len(xyz),
        }

    def collate_fn(self, batch: List[Dict]) -> Dict:
        points_list = [item["points"] for item in batch]
        filenames = [item["filename"] for item in batch]
        return {
            "points": points_list,
            "filenames": filenames,
        }


@dataclass
class nuScenesLidarDatasetFromDir(nuScenesLidarDataset):
    """
    从目录自动扫描 .bin 文件的简化版 Dataset。
    """
    file_list: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        bin_files = sorted(self.data_dir.glob("*.bin"))
        self.valid_files = [f.name for f in bin_files]
        if len(self.valid_files) == 0:
            raise FileNotFoundError(
                f"No .bin files found in {self.data_dir}. "
                f"Please check if the data directory is correct."
            )
