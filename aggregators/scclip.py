"""
SCCLIP Aggregator Implementation
Self-Centered Clipping for Byzantine-Robust Decentralized FL
Reference: He et al. (2022) "Byzantine-Robust Decentralized Learning via Self-Centered Clipping"
"""
import torch
import numpy as np
from collections import OrderedDict
from .base import BaseAggregator


class SCCLIPAggregator(BaseAggregator):
    """
    SCCLIP聚合器（对比基线）

    核心机制:
    1. 计算差向量: d_j = w_j - w_i
    2. 自适应裁剪半径: tau_i = C * sqrt(sum ||w_i - w_j||^2)
    3. 裁剪: clip(d_j) = d_j * min(1, tau / ||d_j||)
    4. 聚合: AGG_i = w_i + (1/|N_i|) * sum clip(d_j)

    特点: 只约束幅度，不检查方向
    """

    def __init__(self, alpha=0.5, clip_constant=0.1):
        """
        参数:
            alpha: 自锚定权重
            clip_constant: 裁剪常数C，控制裁剪半径
        """
        super().__init__(name="SCCLIP", alpha=alpha)
        self.clip_constant = clip_constant

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        """
        SCCLIP聚合逻辑

        参数同SAMA/BALANCE
        """
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'num_clipped': 0}
            return own_model

        w_i_vec = self.model_to_vector(own_model)

        # 计算所有差向量和距离
        diff_vecs = []
        distances = []
        for neighbor in neighbor_models:
            w_j_vec = self.model_to_vector(neighbor)
            d_j = w_j_vec - w_i_vec
            dist = torch.norm(d_j).item()
            diff_vecs.append(d_j)
            distances.append(dist)

        # 自适应裁剪半径: tau = C * sqrt((1/|N|) * sum ||d_j||^2)
        mean_sq_dist = sum(d ** 2 for d in distances) / len(distances)
        tau = self.clip_constant * np.sqrt(mean_sq_dist) if mean_sq_dist > 0 else 1e-8

        # 裁剪每个差向量
        clipped_vecs = []
        num_clipped = 0
        for d_j, dist in zip(diff_vecs, distances):
            if dist > tau and dist > 1e-8:
                d_j_clipped = d_j * (tau / dist)
                num_clipped += 1
            else:
                d_j_clipped = d_j
            clipped_vecs.append(d_j_clipped)

        # 聚合: w_i + mean(clipped diffs)
        avg_clipped = torch.stack(clipped_vecs).mean(dim=0)

        # 数值保护：检查 nan/inf
        if torch.isnan(avg_clipped).any() or torch.isinf(avg_clipped).any():
            if return_stats:
                return own_model, {'num_neighbors': len(neighbor_models), 'num_clipped': num_clipped,
                                  'clip_radius': tau, 'avg_distance': np.mean(distances),
                                  'clip_rate': 0.0, 'avg_trust': None}
            return own_model

        agg_vec = w_i_vec + avg_clipped
        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            clip_rate = num_clipped / len(neighbor_models)
            stats = {
                'num_neighbors': len(neighbor_models),
                'num_clipped': num_clipped,
                'clip_radius': tau,
                'avg_distance': np.mean(distances),
                'clip_rate': clip_rate,
                'avg_trust': 1.0 - clip_rate,
            }
            return agg_model, stats

        return agg_model
