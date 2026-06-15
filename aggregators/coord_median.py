"""
Coordinate-wise Median Aggregator（去中心化版本）
Reference: Yin et al. (2018) "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
"""
import torch
import numpy as np
from .base import BaseAggregator


class CoordMedianAggregator(BaseAggregator):
    """
    坐标中位数聚合器（去中心化版本，对比基线）

    核心机制:
    - 对每个参数维度，计算所有候选值的中位数
    - 无需预设拜占庭比例，天然抵抗不超过 50% 的拜占庭节点

    去中心化适配: 每个节点对其邻居子集（含自身）执行此操作
    """

    def __init__(self, alpha=0.5):
        super().__init__(name="CoordMedian", alpha=alpha)

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'avg_trust': None}
            return own_model

        candidates = [own_model] + neighbor_models
        vecs = torch.stack([self.model_to_vector(m) for m in candidates])  # [n, d]

        agg_vec = vecs.median(dim=0).values
        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            return agg_model, {'num_neighbors': len(neighbor_models), 'avg_trust': None}
        return agg_model
