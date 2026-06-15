"""
Krum / Multi-Krum Aggregator（去中心化版本）
Reference: Blanchard et al. (2017) "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
"""
import torch
import numpy as np
from .base import BaseAggregator


class KrumAggregator(BaseAggregator):
    """
    Krum / Multi-Krum聚合器（去中心化版本，对比基线）

    核心机制:
    - 对每个候选模型 w_j，计算其到最近 (n-f-2) 个邻居的距离平方和作为 Krum 得分
    - Krum: 选得分最低的单个模型
    - Multi-Krum: 选得分最低的 m 个模型取均值

    去中心化适配: n 和 f 基于各节点的邻居子集大小计算
    """

    def __init__(self, alpha=0.5, multi_k=None, byzantine_ratio=0.2):
        """
        参数:
            alpha: 自锚定权重
            multi_k: Multi-Krum 选取的模型数量（None 表示使用 Krum）
            byzantine_ratio: 预期拜占庭比例，用于估算 f
        """
        super().__init__(name="Multi-Krum" if multi_k else "Krum", alpha=alpha)
        self.multi_k = multi_k
        self.byzantine_ratio = byzantine_ratio

    def _krum_scores(self, vecs, f):
        """
        计算每个向量的 Krum 得分
        得分 = 到最近 (n-f-2) 个向量的距离平方和
        """
        n = len(vecs)
        k = max(1, n - f - 2)
        scores = []
        for i, vi in enumerate(vecs):
            dists = sorted(
                [torch.norm(vi - vecs[j]).item() ** 2 for j in range(n) if j != i]
            )
            scores.append(sum(dists[:k]))
        return scores

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'avg_trust': None}
            return own_model

        candidates = [own_model] + neighbor_models
        vecs = [self.model_to_vector(m) for m in candidates]
        n = len(candidates)
        f = max(1, int(n * self.byzantine_ratio))

        scores = self._krum_scores(vecs, f)

        if self.multi_k is not None:
            m = min(self.multi_k, n - f)
            selected_idx = sorted(range(n), key=lambda i: scores[i])[:m]
            agg_vec = torch.stack([vecs[i] for i in selected_idx]).mean(dim=0)
        else:
            best_idx = int(np.argmin(scores))
            agg_vec = vecs[best_idx]

        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            return agg_model, {
                'num_neighbors': len(neighbor_models),
                'avg_trust': None,
                'krum_scores': scores,
            }
        return agg_model
