"""
SCCLIP Aggregator Implementation
"""
import torch
import numpy as np
from .base import BaseAggregator


class SCCLIPAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, clip_constant=0.1):
        super().__init__(name="SCCLIP", alpha=alpha)
        self.clip_constant = clip_constant

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'num_clipped': 0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        diff_vecs = neighbor_mat - own_vec.unsqueeze(0)  # [N, D]
        dists = torch.norm(diff_vecs, dim=1)             # [N]

        mean_sq_dist = (dists ** 2).mean().item()
        tau = self.clip_constant * np.sqrt(mean_sq_dist) if mean_sq_dist > 1e-16 else 1e-8

        clip_scale = torch.clamp(tau / dists.clamp(min=1e-8), max=1.0)
        clipped = diff_vecs * clip_scale.unsqueeze(1)
        num_clipped = int((dists > tau).sum().item())

        avg_clipped = clipped.mean(dim=0)
        if torch.isnan(avg_clipped).any() or torch.isinf(avg_clipped).any():
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_clipped': num_clipped,
                                 'clip_radius': tau, 'avg_distance': dists.mean().item(),
                                 'clip_rate': 0.0, 'avg_trust': None}
            return own_vec

        agg_vec = own_vec + avg_clipped

        if return_stats:
            clip_rate = num_clipped / N
            return agg_vec, {
                'num_neighbors': N, 'num_clipped': num_clipped,
                'clip_radius': tau, 'avg_distance': dists.mean().item(),
                'clip_rate': clip_rate, 'avg_trust': 1.0 - clip_rate,
            }
        return agg_vec
