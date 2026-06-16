"""
Trimmed Mean Aggregator
"""
import torch
from .base import BaseAggregator


class TrimmedMeanAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, trim_ratio=0.1):
        super().__init__(name="TrimmedMean", alpha=alpha)
        self.trim_ratio = trim_ratio

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'avg_trust': None}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        vecs = torch.cat([own_vec.unsqueeze(0), neighbor_mat], dim=0)  # [N, D]
        n = vecs.shape[0]
        k = max(0, int(n * self.trim_ratio))
        if 2 * k >= n:
            k = 0

        sorted_vecs, _ = torch.sort(vecs, dim=0)
        trimmed = sorted_vecs[k:n - k] if k > 0 else sorted_vecs
        agg_vec = trimmed.mean(dim=0)

        if return_stats:
            return agg_vec, {'num_neighbors': n - 1, 'avg_trust': None, 'trim_k': k}
        return agg_vec
