"""
Krum / Multi-Krum Aggregator
"""
import torch
import numpy as np
from .base import BaseAggregator


class KrumAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, multi_k=None, byzantine_ratio=0.2):
        super().__init__(name="Multi-Krum" if multi_k else "Krum", alpha=alpha)
        self.multi_k = multi_k
        self.byzantine_ratio = byzantine_ratio

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
        f = max(1, int(n * self.byzantine_ratio))
        k = max(1, n - f - 2)

        diff = vecs.unsqueeze(0) - vecs.unsqueeze(1)   # [N, N, D]
        dist_sq = (diff ** 2).sum(dim=2)               # [N, N]
        dist_sq.fill_diagonal_(float('inf'))
        topk_vals, _ = torch.topk(dist_sq, k, dim=1, largest=False)
        scores = topk_vals.sum(dim=1)                  # [N]

        if self.multi_k is not None:
            m = min(self.multi_k, n - f)
            selected_idx = torch.topk(scores, m, largest=False).indices
            agg_vec = vecs[selected_idx].mean(dim=0)
        else:
            agg_vec = vecs[scores.argmin()]

        if return_stats:
            return agg_vec, {
                'num_neighbors': n - 1,
                'avg_trust': None,
                'krum_scores': scores.cpu().tolist(),
            }
        return agg_vec
