"""
SAMA-DFL Aggregator Implementation
"""
import torch
import numpy as np
from .base import BaseAggregator


class SAMAAggregator(BaseAggregator):

    def __init__(self, alpha=0.5, tau_max=1.0, tau_min=0.01, use_temperature=False,
                 trust_layers=None, eps=1e-8):
        super().__init__(name="SAMA-DFL", alpha=alpha)
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.use_temperature = use_temperature
        self.trust_layers = trust_layers
        self.eps = eps
        self._trust_indices = None

    def compute_temperature(self, t, T):
        if not self.use_temperature:
            return 1.0
        decay = np.exp(-5.0 * t / T)
        return self.tau_min + (self.tau_max - self.tau_min) * decay

    def _get_trust_indices(self, model):
        if self._trust_indices is not None:
            return self._trust_indices
        if self.trust_layers is None:
            return None
        idx = 0
        slices = []
        state = model.state_dict()
        for key in sorted(state.keys()):
            param = state[key]
            if not isinstance(param, torch.Tensor):
                continue
            n = param.numel()
            if key in self.trust_layers:
                slices.append((idx, idx + n))
            idx += n
        self._trust_indices = slices
        return slices

    def _extract_trust_vec(self, full_vec, model=None):
        if self.trust_layers is None or model is None:
            return full_vec
        slices = self._get_trust_indices(model)
        if not slices:
            return full_vec
        return torch.cat([full_vec[s:e] for s, e in slices])

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False,
                  model=None, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'num_filtered': 0, 'avg_trust': 0.0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        w_i_norm = torch.norm(own_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_vec, {'error': 'own_model_zero_norm'}
            return own_vec

        w_i_trust = self._extract_trust_vec(own_vec, model)
        w_i_trust_norm = torch.norm(w_i_trust)

        neighbor_trust = torch.stack([self._extract_trust_vec(neighbor_mat[i], model)
                                       for i in range(N)])
        neighbor_trust_norms = torch.norm(neighbor_trust, dim=1)
        neighbor_norms = torch.norm(neighbor_mat, dim=1)

        tau = self.compute_temperature(t, T)

        valid_mask = (neighbor_norms >= self.eps) & (neighbor_trust_norms >= self.eps) & (w_i_trust_norm >= self.eps)
        cos_sims = torch.zeros(N, device=own_vec.device)
        if valid_mask.any():
            dots = (neighbor_trust[valid_mask] * w_i_trust).sum(dim=1)
            cos_sims[valid_mask] = dots / (neighbor_trust_norms[valid_mask] * w_i_trust_norm)

        phi = torch.clamp(cos_sims / tau if self.use_temperature else cos_sims, min=0.0)
        valid = (phi > 0) & valid_mask

        if not valid.any():
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': N, 'avg_trust': 0.0, 'tau': tau}
            return own_vec

        valid_vecs = neighbor_mat[valid]
        valid_norms = neighbor_norms[valid]
        valid_phi = phi[valid]

        aligned = w_i_norm * (valid_vecs / valid_norms.unsqueeze(1))
        total_weight = valid_phi.sum() + self.eps
        agg_vec = (valid_phi.unsqueeze(1) * aligned).sum(dim=0) / total_weight

        if return_stats:
            phi_vals = valid_phi.cpu().float()
            stats = {
                'num_neighbors': N,
                'num_filtered': N - int(valid.sum().item()),
                'avg_trust': float(phi_vals.mean()),
                'min_trust': float(phi_vals.min()),
                'max_trust': float(phi_vals.max()),
                'tau': tau,
            }
            return agg_vec, stats
        return agg_vec
