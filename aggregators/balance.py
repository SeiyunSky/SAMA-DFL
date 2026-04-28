"""
BALANCE Aggregator Implementation
基于时变阈值筛选的拜占庭鲁棒聚合器
"""
import torch
import numpy as np
from collections import OrderedDict
from .base import BaseAggregator


class BALANCEAggregator(BaseAggregator):
    """
    BALANCE聚合器（对比基线）

    核心机制:
    1. 相似性检查: ||w_i - w_j|| ≤ γ·exp(-κ·t/T)·||w_i||
    2. 硬阈值筛选: 通过则保留，否则丢弃
    3. 自锚定融合: w_i^{t+1} = α·w_i' + (1-α)·(1/|S|)·Σw_j
    """

    def __init__(self, alpha=0.5, gamma=0.5, kappa=0.1):
        """
        参数:
            alpha: 自锚定权重
            gamma: 基础阈值系数
            kappa: 衰减速率
        """
        super().__init__(name="BALANCE")
        self.alpha = alpha
        self.gamma = gamma
        self.kappa = kappa

    def compute_threshold(self, t, T, own_norm):
        """
        计算时变阈值
        T(t) = γ · exp(-κ·t/T) · ||w_i||
        """
        decay = np.exp(-self.kappa * t / T)
        return self.gamma * decay * own_norm

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        """
        BALANCE聚合逻辑

        参数同SAMA
        """
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'num_accepted': 0}
            return own_model

        # 转换为向量
        w_i_vec = self.model_to_vector(own_model)
        w_i_norm = torch.norm(w_i_vec).item()

        # 计算阈值
        threshold = self.compute_threshold(t, T, w_i_norm)

        # 硬筛选
        accepted = []
        distances = []

        for neighbor in neighbor_models:
            w_j_vec = self.model_to_vector(neighbor)
            dist = torch.norm(w_i_vec - w_j_vec).item()
            distances.append(dist)

            if dist <= threshold:
                accepted.append(neighbor)

        if not accepted:
            # 无邻居通过，返回自身
            if return_stats:
                stats = {
                    'num_neighbors': len(neighbor_models),
                    'num_accepted': 0,
                    'threshold': threshold,
                    'avg_distance': np.mean(distances)
                }
                return own_model, stats
            return own_model

        # 平均聚合通过的邻居
        agg_vec = torch.stack([self.model_to_vector(m) for m in accepted]).mean(dim=0)
        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            stats = {
                'num_neighbors': len(neighbor_models),
                'num_accepted': len(accepted),
                'threshold': threshold,
                'avg_distance': np.mean(distances),
                'acceptance_rate': len(accepted) / len(neighbor_models)
            }
            return agg_model, stats

        return agg_model

    def final_update(self, local_model, aggregated_model, alpha=None, **kwargs):
        """自锚定融合"""
        if alpha is None:
            alpha = self.alpha

        final_state = OrderedDict()
        for key in local_model.keys():
            if isinstance(local_model[key], torch.Tensor):
                final_state[key] = (alpha * local_model[key] +
                                   (1 - alpha) * aggregated_model[key])
            else:
                final_state[key] = local_model[key]

        return final_state
