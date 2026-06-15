"""
FedAvg Aggregator（去中心化版本）
每个节点对其邻居子集做简单均值，无拜占庭防御
"""
import torch
import numpy as np
from .base import BaseAggregator


class FedAvgAggregator(BaseAggregator):
    """
    FedAvg聚合器（去中心化版本，对比基线）

    核心机制: AGG_i = (1/|N_i|) · Σ w_j
    无任何拜占庭过滤，作为无防御基线
    """

    def __init__(self, alpha=0.5):
        super().__init__(name="FedAvg", alpha=alpha)

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'avg_trust': None}
            return own_model

        all_models = [own_model] + neighbor_models
        agg_vec = torch.stack([self.model_to_vector(m) for m in all_models]).mean(dim=0)
        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            return agg_model, {'num_neighbors': len(neighbor_models), 'avg_trust': None}
        return agg_model
