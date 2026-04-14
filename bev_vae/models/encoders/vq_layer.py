import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.cluster.vq import kmeans2
import numpy as np


class VectorQuantizer(nn.Module):
    """
    VQ-VAE 的核心：向量量化层。
    
    输入：连续的特征向量 (B, L, D)
    输出：量化后的向量 (B, L, D) + 量化损失 + 码本索引 (B, L)
    """
    def __init__(self, n_e=1024, e_dim=1024, beta=1.0, cosine_similarity=False, dead_limit=256):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.cosine_similarity = cosine_similarity
        self.dead_limit = dead_limit

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        self.register_buffer("num_iter", torch.zeros(1))
        self.register_buffer("data_initialized", torch.zeros(1))
        self.register_buffer("reservoir", torch.zeros(self.n_e * 10, e_dim))

    def forward(self, z, code_age=None, code_usage=None):
        assert z.shape[-1] == self.e_dim
        z_flattened = z.reshape(-1, self.e_dim)

        if self.cosine_similarity:
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)

        self._update_reservoir(z_flattened, code_age, code_usage)

        if self.cosine_similarity:
            min_encoding_indices = torch.matmul(
                z_flattened, F.normalize(self.embedding.weight, p=2, dim=-1).T
            ).max(dim=-1)[1]
        else:
            z_dist = torch.cdist(z_flattened, self.embedding.weight)
            min_encoding_indices = torch.argmin(z_dist, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)

        if self.cosine_similarity:
            z_q = F.normalize(z_q, p=2, dim=-1)
            z_norm = F.normalize(z, p=2, dim=-1)
            loss = (
                self.beta * torch.mean(1 - (z_q.detach() * z_norm).sum(dim=-1)),
                torch.mean(1 - (z_q * z_norm.detach()).sum(dim=-1)),
            )
        else:
            loss = (
                self.beta * torch.mean((z_q.detach() - z) ** 2),
                torch.mean((z_q - z.detach()) ** 2),
            )

        z_q = z + (z_q - z).detach()

        if code_age is not None and code_usage is not None:
            code_idx = min_encoding_indices
            if torch.distributed.is_initialized():
                code_idx = torch.cat(torch.distributed.nn.functional.all_gather(code_idx))
            code_age += 1
            code_age[code_idx] = 0
            code_usage.index_add_(0, code_idx, torch.ones_like(code_idx, dtype=code_usage.dtype))

        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None):
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        if self.cosine_similarity:
            z_q = F.normalize(z_q, p=2, dim=-1)
        return z_q

    def _update_reservoir(self, z, code_age, code_usage):
        if not (self.embedding.weight.requires_grad and self.training):
            return
        z_flattened = z.reshape(-1, self.e_dim)
        rp = torch.randperm(z_flattened.size(0))
        num_sample = self.reservoir.shape[0] // 100
        self.reservoir = torch.cat([self.reservoir[num_sample:], z_flattened[rp[:num_sample]].data])
        self.num_iter += 1

    def _update_codebook(self, code_age, code_usage):
        if not (self.embedding.weight.requires_grad and self.training):
            return
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            live_code = self.embedding.weight[code_age < self.dead_limit].data
            live_code_num = live_code.shape[0]
            if self.cosine_similarity:
                live_code = F.normalize(live_code, p=2, dim=-1)
            all_z = torch.cat([self.reservoir, live_code])
            rp = torch.randperm(all_z.shape[0])
            all_z = all_z[rp]
            init = torch.cat(
                live_code,
                self.reservoir[torch.randperm(self.reservoir.shape[0])[: (self.n_e - live_code_num)]]
            )
            init = init.data.cpu().numpy()
            centroid, assignment = kmeans2(
                all_z.cpu().numpy(), init, minit="matrix", iter=50,
            )
            self.embedding.weight.data = torch.from_numpy(centroid).to(self.embedding.weight.device)
        if torch.distributed.is_initialized():
            torch.distributed.nn.functional.broadcast(self.embedding.weight, src=0)
        code_age.fill_(0)
        code_usage.fill_(0)
