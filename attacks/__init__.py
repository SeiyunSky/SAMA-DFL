"""
Byzantine attack implementations
"""
import torch
import numpy as np
from collections import OrderedDict


class NoAttack:
    """无攻击基线，拜占庭节点正常训练"""
    def __init__(self):
        self.name = "NoAttack"

    def attack(self, model_state):
        return model_state


class GaussianAttack:
    """
    高斯攻击: 用随机高斯噪声替换梯度
    g_mal ~ N(0, σ²)
    """
    def __init__(self, std=10.0):
        self.std = std
        self.name = "Gaussian"

    def attack(self, model_state):
        malicious_state = OrderedDict()
        for key, param in model_state.items():
            if isinstance(param, torch.Tensor):
                noise = torch.randn_like(param) * self.std
                malicious_state[key] = param + noise
            else:
                malicious_state[key] = param
        return malicious_state


class LabelFlippingAttack:
    """
    标签翻转攻击: 在本地训练时翻转标签
    y_mal = (num_classes - 1) - y_true
    """
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.name = "LabelFlipping"

    def flip_labels(self, labels):
        return (self.num_classes - 1) - labels

    def attack_dataset(self, dataset):
        pass


class OmniscientAttack:
    """
    全知攻击: 利用已知的诚实梯度之和，发送其负值
    g_mal = -C · mean(g_honest)
    """
    def __init__(self, amplification=2.0):
        self.amplification = amplification
        self.name = "Omniscient"

    def attack(self, honest_models):
        if not honest_models:
            return None

        avg_state = OrderedDict()
        for key in honest_models[0].keys():
            if isinstance(honest_models[0][key], torch.Tensor):
                stacked = torch.stack([m[key] for m in honest_models])
                avg_state[key] = stacked.mean(dim=0)
            else:
                avg_state[key] = honest_models[0][key]

        malicious_state = OrderedDict()
        for key, param in avg_state.items():
            if isinstance(param, torch.Tensor):
                malicious_state[key] = -self.amplification * param
            else:
                malicious_state[key] = param

        return malicious_state


class KrumAttack:
    """
    Krum 攻击 (Fang et al., 2020)
    针对 Krum/Multi-Krum 防御的自适应攻击。

    核心思路:
    - Krum 选"到最近 (n-f-2) 个邻居距离平方和最小"的模型
    - 攻击者沿攻击方向 v = -mean(honest) 构造恶意模型
    - 二分搜索步长 λ，使恶意模型的 Krum 得分 < 所有诚实模型的 Krum 得分
    - 若搜索失败（防御太强），退化为小幅 Omniscient 攻击

    Reference: Fang et al. "Local Model Poisoning Attacks to Byzantine-Robust FL" (USENIX Security 2020)
    """

    def __init__(self, num_byzantine=4, amplification=1.0):
        """
        参数:
            num_byzantine: 拜占庭节点总数 f
            amplification: 攻击方向的初始步长倍数
        """
        self.num_byzantine = num_byzantine
        self.amplification = amplification
        self.name = "KrumAttack"

    def _flatten(self, state):
        return torch.cat([p.reshape(-1).float() for p in state.values()
                          if isinstance(p, torch.Tensor)])

    def _unflatten(self, vec, ref_state):
        result = OrderedDict()
        idx = 0
        for key, param in ref_state.items():
            if isinstance(param, torch.Tensor):
                n = param.numel()
                result[key] = vec[idx:idx + n].reshape(param.shape)
                idx += n
            else:
                result[key] = param
        return result

    def _krum_score(self, target_idx, all_vecs, f):
        """计算 all_vecs[target_idx] 对其余向量的 Krum 得分"""
        n = len(all_vecs)
        k = max(1, n - f - 2)
        target_vec = all_vecs[target_idx]
        dists = sorted([torch.norm(target_vec - all_vecs[j]).item() ** 2
                        for j in range(n) if j != target_idx])
        return sum(dists[:k])

    def attack(self, honest_models):
        f = self.num_byzantine
        honest_vecs = [self._flatten(m) for m in honest_models]
        mean_honest = torch.stack(honest_vecs).mean(dim=0)

        # 攻击方向：诚实均值的反方向
        attack_dir = -mean_honest
        attack_dir_norm = torch.norm(attack_dir)
        if attack_dir_norm < 1e-8:
            attack_dir = torch.randn_like(mean_honest)
            attack_dir_norm = torch.norm(attack_dir)
        attack_dir = attack_dir / attack_dir_norm

        # 诚实模型中最小的 Krum 得分（攻击目标：低于这个值）
        honest_scores = [self._krum_score(i, honest_vecs, f) for i in range(len(honest_vecs))]
        target_score = min(honest_scores)

        # 二分搜索步长 λ
        lo, hi = 0.0, torch.norm(mean_honest).item() * self.amplification * 10
        mal_vec = mean_honest.clone()

        for _ in range(50):
            mid = (lo + hi) / 2.0
            candidate = mean_honest + mid * attack_dir
            all_vecs = honest_vecs + [candidate]
            score = self._krum_score(len(all_vecs) - 1, all_vecs, f)
            if score < target_score:
                lo = mid
                mal_vec = candidate
            else:
                hi = mid

        return self._unflatten(mal_vec, honest_models[0])


class TrimAttack:
    """
    Trim 攻击 (Fang et al., 2020)
    针对 Trimmed Mean 防御的自适应攻击。

    核心思路:
    - Trimmed Mean 对每个维度排序后裁掉两端各 β 比例
    - 攻击者对每个维度，把恶意值推到"刚好不被裁掉"的边界
    - 攻击方向 v = -mean(honest)（使模型偏离正确方向）
    - 对每个维度 d：
        - 若 v[d] < 0：把恶意值设为排序后第 β 个位置的值（下边界），尽量小
        - 若 v[d] > 0：把恶意值设为排序后第 (1-β) 个位置的值（上边界），尽量大
    - 这样恶意值恰好落在保留区间的边界，不被裁掉但最大化偏移

    Reference: Fang et al. "Local Model Poisoning Attacks to Byzantine-Robust FL" (USENIX Security 2020)
    """

    def __init__(self, num_byzantine=4, trim_ratio=0.1):
        """
        参数:
            num_byzantine: 拜占庭节点总数 f
            trim_ratio: Trimmed Mean 的裁剪比例 β（需与防御方一致）
        """
        self.num_byzantine = num_byzantine
        self.trim_ratio = trim_ratio
        self.name = "TrimAttack"

    def _flatten(self, state):
        return torch.cat([p.reshape(-1).float() for p in state.values()
                          if isinstance(p, torch.Tensor)])

    def _unflatten(self, vec, ref_state):
        result = OrderedDict()
        idx = 0
        for key, param in ref_state.items():
            if isinstance(param, torch.Tensor):
                n = param.numel()
                result[key] = vec[idx:idx + n].reshape(param.shape)
                idx += n
            else:
                result[key] = param
        return result

    def attack(self, honest_models):
        honest_vecs = torch.stack([self._flatten(m) for m in honest_models])  # [h, d]
        mean_honest = honest_vecs.mean(dim=0)  # [d]

        # 攻击方向：每个维度独立推向反方向
        attack_sign = -torch.sign(mean_honest)

        n_honest = honest_vecs.shape[0]
        n_total = n_honest + self.num_byzantine
        # 防御方裁剪边界索引（每端裁 k 个）
        k = max(1, int(n_total * self.trim_ratio))

        sorted_vals, _ = torch.sort(honest_vecs, dim=0)  # [h, d]

        # 目标：恶意值落在保留区间内，但尽量靠近边界以最大化偏移
        # 对 attack_sign < 0（推低）：取诚实节点排序后第 k 个值（保留区下边界）再减去 γ·std
        # 对 attack_sign > 0（推高）：取诚实节点排序后第 -k-1 个值（保留区上边界）再加上 γ·std
        # 注意：此处的 sorted_vals 仅含诚实节点，实际排序还会含 num_byzantine 个恶意值
        # 使用更激进的偏移以确保攻击有效
        std = honest_vecs.std(dim=0).clamp(min=1e-6)
        gamma = 10.0

        lower_idx = min(k, n_honest - 1)
        upper_idx = max(n_honest - 1 - k, 0)
        lower_bound = sorted_vals[lower_idx] - gamma * std
        upper_bound = sorted_vals[upper_idx] + gamma * std

        mal_vec = torch.where(attack_sign < 0, lower_bound, upper_bound)

        return self._unflatten(mal_vec, honest_models[0])


# 攻击工厂
ATTACK_REGISTRY = {
    'none': NoAttack,
    'gaussian': GaussianAttack,
    'label_flipping': LabelFlippingAttack,
    'omniscient': OmniscientAttack,
    'krum_attack': KrumAttack,
    'trim_attack': TrimAttack,
}


def get_attack(attack_name, **kwargs):
    """获取攻击实例"""
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}")
    return ATTACK_REGISTRY[attack_name](**kwargs)
