"""
Trimmed Mean Aggregator（去中心化版本）
Reference: Yin et al. (2018) "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
"""
import torch
import numpy as np
from .base import BaseAggregator


class TrimmedMeanAggregator(BaseAggregator):
    """
    坐标裁剪均值聚合器（去中心化版本，对比基线）

    核心机制:
    - 对每个参数维度，将所有候选值排序
    - 去掉最大的 β 比例和最小的 β 比例
    - 对剩余值取均值

    去中心化适配: 每个节点对其邻居子集（含自身）执行此操作
    """

    def __init__(self, alpha=0.5, trim_ratio=0.1):
        """
        参数:
            alpha: 自锚定权重
            trim_ratio: 每端裁剪比例 β（0.1 表示各裁掉 10%）
        """
        super().__init__(name="TrimmedMean", alpha=alpha)
        self.trim_ratio = trim_ratio

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'avg_trust': None}
            return own_model

        candidates = [own_model] + neighbor_models
        vecs = torch.stack([self.model_to_vector(m) for m in candidates])  # [n, d]
        n = vecs.shape[0]

        k = max(1, int(n * self.trim_ratio))
        if 2 * k >= n:
            k = 0

        sorted_vecs, _ = torch.sort(vecs, dim=0)

        if k > 0:
            trimmed = sorted_vecs[k:n - k, :]
        else:
            trimmed = sorted_vecs

        agg_vec = trimmed.mean(dim=0)
        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            return agg_model, {
                'num_neighbors': len(neighbor_models),
                'avg_trust': None,
                'trim_k': k,
            }
        return agg_model
