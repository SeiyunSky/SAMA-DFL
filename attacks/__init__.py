"""
Byzantine attack implementations — 全部使用 GPU tensor 接口
"""
import torch
import numpy as np


class NoAttack:
    def __init__(self):
        self.name = "NoAttack"

    def attack(self, own_vec):
        return own_vec


class GaussianAttack:
    def __init__(self, std=10.0):
        self.std = std
        self.name = "Gaussian"

    def attack(self, own_vec):
        return own_vec + torch.randn_like(own_vec) * self.std


class LabelFlippingAttack:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.name = "LabelFlipping"

    def flip_labels(self, labels):
        return (self.num_classes - 1) - labels


class OmniscientAttack:
    def __init__(self, amplification=2.0):
        self.amplification = amplification
        self.name = "Omniscient"

    def attack(self, honest_vecs):
        """honest_vecs: List[Tensor[D]] 或 Tensor[N, D]"""
        if isinstance(honest_vecs, list):
            if not honest_vecs:
                return None
            mat = torch.stack(honest_vecs)
        else:
            mat = honest_vecs
        return -self.amplification * mat.mean(dim=0)


class KrumAttack:
    def __init__(self, num_byzantine=4, amplification=1.0):
        self.num_byzantine = num_byzantine
        self.amplification = amplification
        self.name = "KrumAttack"

    def _krum_score(self, target_idx, all_vecs, f):
        n = len(all_vecs)
        k = max(1, n - f - 2)
        target_vec = all_vecs[target_idx]
        dists = sorted([torch.norm(target_vec - all_vecs[j]).item() ** 2
                        for j in range(n) if j != target_idx])
        return sum(dists[:k])

    def attack(self, honest_vecs):
        """honest_vecs: List[Tensor[D]]"""
        if isinstance(honest_vecs, list):
            mat = torch.stack(honest_vecs)
        else:
            mat = honest_vecs
        f = self.num_byzantine
        mean_honest = mat.mean(dim=0)

        attack_dir = -mean_honest
        attack_dir_norm = torch.norm(attack_dir)
        if attack_dir_norm < 1e-8:
            attack_dir = torch.randn_like(mean_honest)
            attack_dir_norm = torch.norm(attack_dir)
        attack_dir = attack_dir / attack_dir_norm

        vecs_list = [mat[i] for i in range(mat.shape[0])]
        honest_scores = [self._krum_score(i, vecs_list, f) for i in range(len(vecs_list))]
        target_score = min(honest_scores)

        lo, hi = 0.0, torch.norm(mean_honest).item() * self.amplification * 10
        mal_vec = mean_honest.clone()

        for _ in range(50):
            mid = (lo + hi) / 2.0
            candidate = mean_honest + mid * attack_dir
            all_vecs = vecs_list + [candidate]
            score = self._krum_score(len(all_vecs) - 1, all_vecs, f)
            if score < target_score:
                lo = mid
                mal_vec = candidate
            else:
                hi = mid

        return mal_vec


class TrimAttack:
    def __init__(self, num_byzantine=4, trim_ratio=0.1):
        self.num_byzantine = num_byzantine
        self.trim_ratio = trim_ratio
        self.name = "TrimAttack"

    def attack(self, honest_vecs):
        """honest_vecs: List[Tensor[D]]"""
        if isinstance(honest_vecs, list):
            mat = torch.stack(honest_vecs)
        else:
            mat = honest_vecs

        mean_honest = mat.mean(dim=0)
        attack_sign = -torch.sign(mean_honest)

        n_honest = mat.shape[0]
        n_total = n_honest + self.num_byzantine
        k = max(1, int(n_total * self.trim_ratio))

        sorted_vals, _ = torch.sort(mat, dim=0)
        std = mat.std(dim=0).clamp(min=1e-6)
        gamma = 10.0

        lower_idx = min(k, n_honest - 1)
        upper_idx = max(n_honest - 1 - k, 0)
        lower_bound = sorted_vals[lower_idx] - gamma * std
        upper_bound = sorted_vals[upper_idx] + gamma * std

        return torch.where(attack_sign < 0, lower_bound, upper_bound)


ATTACK_REGISTRY = {
    'none': NoAttack,
    'gaussian': GaussianAttack,
    'label_flipping': LabelFlippingAttack,
    'omniscient': OmniscientAttack,
    'krum_attack': KrumAttack,
    'trim_attack': TrimAttack,
}


def get_attack(attack_name, **kwargs):
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}")
    return ATTACK_REGISTRY[attack_name](**kwargs)
