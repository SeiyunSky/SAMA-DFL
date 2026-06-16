"""
BALANCE Aggregator Implementation
"""
import torch
import numpy as np
from .base import BaseAggregator


class BALANCEAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, gamma=0.5, kappa=0.1):
        super().__init__(name="BALANCE", alpha=alpha)
        self.gamma = gamma
        self.kappa = kappa

    def compute_threshold(self, t, T, own_norm):
        decay = np.exp(-self.kappa * t / T)
        return self.gamma * decay * own_norm

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'num_accepted': 0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        w_i_norm = torch.norm(own_vec).item()
        threshold = self.compute_threshold(t, T, w_i_norm)

        dists = torch.norm(neighbor_mat - own_vec.unsqueeze(0), dim=1)
        accept_mask = dists <= threshold

        if not accept_mask.any():
            if return_stats:
                return own_vec, {
                    'num_neighbors': N, 'num_accepted': 0,
                    'threshold': threshold,
                    'avg_distance': dists.mean().item(),
                    'avg_trust': 0.0,
                }
            return own_vec

        agg_vec = neighbor_mat[accept_mask].mean(dim=0)

        if return_stats:
            num_accepted = int(accept_mask.sum().item())
            accept_rate = num_accepted / N
            return agg_vec, {
                'num_neighbors': N, 'num_accepted': num_accepted,
                'threshold': threshold,
                'avg_distance': dists.mean().item(),
                'acceptance_rate': accept_rate,
                'avg_trust': accept_rate,
            }
        return agg_vec
