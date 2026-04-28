"""
SAMA-DFL Aggregator Implementation
基于自锚定幅度对齐的拜占庭鲁棒去中心化联邦学习聚合器
"""
import torch
import numpy as np
from collections import OrderedDict
from .base import BaseAggregator


class SAMAAggregator(BaseAggregator):
    """
    SAMA-DFL核心聚合器

    四步机制:
    1. 方向信任权重计算: φ_j = ReLU(cos(w_i, w_j))
    2. 幅度对齐: w̃_j = ||w_i|| · (w_j / ||w_j||)
    3. 软加权聚合: AGG_i = Σ(φ_j · w̃_j) / Σφ_k
    4. 自锚定融合: w_i^{t+1} = α·w_i' + (1-α)·AGG_i
    """

    def __init__(self, alpha=0.5, tau_max=1.0, tau_min=0.01, use_temperature=False,
                 trust_layers=None, adaptive_alpha=False, eps=1e-8):
        """
        参数:
            alpha: 自锚定权重 (默认0.5，理论最优)
            tau_max: 温度参数最大值 (可选，默认1.0)
            tau_min: 温度参数最小值 (可选，默认0.01)
            use_temperature: 是否使用温度退火 (默认False)
            trust_layers: 用于计算方向信任的层名列表 (默认None=全部参数)
                          例如 ['fc2.weight', 'fc2.bias'] 只用分类头计算余弦相似度
            adaptive_alpha: 是否根据信任分自适应调整α (默认False)
                           高威胁(低信任)→增大α，低威胁(高信任)→减小α
            eps: 数值稳定性参数
        """
        super().__init__(name="SAMA-DFL")
        self.alpha = alpha
        self.tau_max = tau_max
        self.tau_min = tau_min
        self.use_temperature = use_temperature
        self.trust_layers = trust_layers
        self.adaptive_alpha = adaptive_alpha
        self.eps = eps

    def compute_temperature(self, t, T):
        """
        计算时变温度参数
        τ(t) = τ_min + (τ_max - τ_min) · exp(-5t/T)
        """
        if not self.use_temperature:
            return 1.0
        decay = np.exp(-5.0 * t / T)
        return self.tau_min + (self.tau_max - self.tau_min) * decay

    def _extract_trust_vector(self, state_dict):
        """提取用于计算方向信任的参数子向量"""
        if self.trust_layers is None:
            return self.model_to_vector(state_dict)
        parts = []
        for key in sorted(state_dict.keys()):
            if key in self.trust_layers and isinstance(state_dict[key], torch.Tensor):
                parts.append(state_dict[key].float().flatten())
        if not parts:
            return self.model_to_vector(state_dict)
        return torch.cat(parts)

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        """
        SAMA-DFL核心聚合逻辑

        参数:
            own_model: OrderedDict - 自身本地训练后的模型
            neighbor_models: List[OrderedDict] - 邻居模型列表
            t: int - 当前轮次
            T: int - 总轮次
            return_stats: bool - 是否返回统计信息

        返回:
            OrderedDict - 聚合后的模型 (如果return_stats=False)
            (OrderedDict, dict) - (聚合模型, 统计信息) (如果return_stats=True)
        """
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'num_filtered': 0}
            return own_model

        # Step 0: 转换为向量
        w_i_vec = self.model_to_vector(own_model)
        w_i_norm = torch.norm(w_i_vec)

        if w_i_norm < self.eps:
            if return_stats:
                return own_model, {'error': 'own_model_zero_norm'}
            return own_model

        neighbor_vecs = [self.model_to_vector(model) for model in neighbor_models]

        # 用于方向信任的子向量（可能只是分类层）
        w_i_trust = self._extract_trust_vector(own_model)
        w_i_trust_norm = torch.norm(w_i_trust)
        neighbor_trust_vecs = [self._extract_trust_vector(model) for model in neighbor_models]

        # Step 1: 计算方向信任权重
        tau = self.compute_temperature(t, T)

        trust_scores = []
        aligned_vecs = []

        for idx, w_j_vec in enumerate(neighbor_vecs):
            w_j_norm = torch.norm(w_j_vec)

            if w_j_norm < self.eps:
                trust_scores.append(0.0)
                aligned_vecs.append(None)
                continue

            # 用 trust 子向量计算余弦相似度
            w_j_trust = neighbor_trust_vecs[idx]
            w_j_trust_norm = torch.norm(w_j_trust)

            if w_j_trust_norm < self.eps or w_i_trust_norm < self.eps:
                trust_scores.append(0.0)
                aligned_vecs.append(None)
                continue

            cos_sim = torch.dot(w_i_trust, w_j_trust) / (w_i_trust_norm * w_j_trust_norm)
            cos_sim = cos_sim.item()

            # 应用ReLU(cos) / tau
            if self.use_temperature:
                phi_j = max(0.0, cos_sim / tau)
            else:
                phi_j = max(0.0, cos_sim)

            trust_scores.append(phi_j)

            # Step 2: 幅度对齐（只对通过筛选的邻居）
            if phi_j > 0:
                w_tilde_j = w_i_norm * (w_j_vec / w_j_norm)
                aligned_vecs.append(w_tilde_j)
            else:
                aligned_vecs.append(None)

        # Step 3: 软加权聚合
        valid_indices = [i for i, score in enumerate(trust_scores) if score > 0]

        if not valid_indices:
            # 所有邻居被过滤，返回自身
            if return_stats:
                stats = {
                    'num_neighbors': len(neighbor_models),
                    'num_filtered': len(neighbor_models),
                    'avg_trust': 0.0,
                    'tau': tau
                }
                return own_model, stats
            return own_model

        valid_scores = [trust_scores[i] for i in valid_indices]
        valid_aligned = [aligned_vecs[i] for i in valid_indices]

        # 归一化权重并加权平均
        total_weight = sum(valid_scores) + self.eps
        agg_vec = sum([score * vec for score, vec in zip(valid_scores, valid_aligned)]) / total_weight

        # 转换回模型格式
        agg_model = self.vector_to_model(agg_vec, own_model)

        # 统计信息
        if return_stats:
            stats = {
                'num_neighbors': len(neighbor_models),
                'num_filtered': len(neighbor_models) - len(valid_indices),
                'avg_trust': np.mean(valid_scores),
                'min_trust': np.min(valid_scores),
                'max_trust': np.max(valid_scores),
                'tau': tau
            }
            return agg_model, stats

        return agg_model

    def final_update(self, local_model, aggregated_model, alpha=None, avg_trust=None):
        """
        自锚定融合
        w_i^{t+1} = α·w_i' + (1-α)·AGG_i

        参数:
            local_model: OrderedDict - 本地训练后的模型
            aggregated_model: OrderedDict - 聚合后的模型
            alpha: float - 自锚权重（如果为None则使用初始化的值）
            avg_trust: float - 平均信任分（用于自适应α，可选）

        返回:
            OrderedDict - 融合后的最终模型
        """
        if alpha is None:
            alpha = self.alpha

        # 自适应α：信任低时更信任自身，信任高时更信任邻居
        if self.adaptive_alpha and avg_trust is not None:
            # avg_trust ∈ [0, 1]，映射到 α ∈ [0.3, 0.8]
            # 高信任(1.0) → α=0.3（多用邻居），低信任(0.0) → α=0.8（多用自身）
            alpha = 0.8 - 0.5 * avg_trust

        final_state = OrderedDict()
        for key in local_model.keys():
            if isinstance(local_model[key], torch.Tensor):
                final_state[key] = (alpha * local_model[key] +
                                   (1 - alpha) * aggregated_model[key])
            else:
                final_state[key] = local_model[key]

        return final_state


# 使用示例
if __name__ == "__main__":
    # 测试代码
    print("SAMA-DFL Aggregator Loaded Successfully")

    # 创建简单测试
    dim = 100
    own_model = OrderedDict({'weight': torch.randn(dim)})
    neighbors = [
        OrderedDict({'weight': torch.randn(dim)}) for _ in range(5)
    ]

    aggregator = SAMAAggregator(alpha=0.5, use_temperature=False)

    # 测试聚合
    agg_result, stats = aggregator.aggregate(own_model, neighbors,
                                             t=0, T=100, return_stats=True)
    print(f"\n聚合统计: {stats}")

    # 测试融合
    final_model = aggregator.final_update(own_model, agg_result)
    print(f"最终模型keys: {list(final_model.keys())}")
