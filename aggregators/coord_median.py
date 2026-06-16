"""
Coordinate-wise Median Aggregator
"""
import torch
from .base import BaseAggregator


class CoordMedianAggregator(BaseAggregator):

    def __init__(self, alpha=0.5):
        super().__init__(name="CoordMedian", alpha=alpha)

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'avg_trust': None}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        vecs = torch.cat([own_vec.unsqueeze(0), neighbor_mat], dim=0)
        agg_vec = vecs.median(dim=0).values

        if return_stats:
            return agg_vec, {'num_neighbors': vecs.shape[0] - 1, 'avg_trust': None}
        return agg_vec
