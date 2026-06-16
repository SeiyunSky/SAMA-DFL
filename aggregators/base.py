"""
Base aggregator class for federated learning
"""
from abc import ABC, abstractmethod
from collections import OrderedDict
import torch


class BaseAggregator(ABC):
    """基础聚合器抽象类"""

    def __init__(self, name="BaseAggregator", alpha=0.5):
        self.name = name
        self.alpha = alpha

    # ──────────────────────────────────────────────────────────
    # 模型 <-> 向量 转换（保持向后兼容）
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def model_to_vector(model_or_state):
        """
        将模型或 state_dict 扁平化为 GPU tensor 向量。
        接受 nn.Module 或 OrderedDict 两种输入。
        """
        if isinstance(model_or_state, torch.nn.Module):
            return torch.cat([p.data.reshape(-1).float()
                              for p in model_or_state.parameters()])
        # state_dict
        vectors = []
        for key in sorted(model_or_state.keys()):
            param = model_or_state[key]
            if isinstance(param, torch.Tensor):
                vectors.append(param.reshape(-1).float())
        return torch.cat(vectors)

    @staticmethod
    def load_from_vector(model, vector):
        """将向量直接写回模型参数（in-place，无 CPU 拷贝）。"""
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(vector[idx:idx + n].reshape(p.shape))
            idx += n

    @staticmethod
    def vector_to_model(vector, reference_state_dict):
        """将向量恢复为 state_dict（供攻击模块等旧接口使用）。"""
        new_state = OrderedDict()
        idx = 0
        for key in sorted(reference_state_dict.keys()):
            param = reference_state_dict[key]
            if isinstance(param, torch.Tensor):
                param_numel = param.numel()
                new_state[key] = vector[idx:idx + param_numel].reshape(param.shape)
                idx += param_numel
            else:
                new_state[key] = param
        return new_state

    def final_update(self, own_vec_or_model, agg_vec_or_model, alpha=None, **kwargs):
        """
        自锚定融合: w^{t+1} = α·w' + (1-α)·AGG
        接受向量或 state_dict，返回与输入相同类型。
        """
        if alpha is None:
            alpha = self.alpha

        # 纯向量模式（新路径）
        if isinstance(own_vec_or_model, torch.Tensor):
            return alpha * own_vec_or_model + (1 - alpha) * agg_vec_or_model

        # state_dict 模式（旧路径，向后兼容）
        final_state = OrderedDict()
        for key in own_vec_or_model.keys():
            if isinstance(own_vec_or_model[key], torch.Tensor):
                final_state[key] = (alpha * own_vec_or_model[key]
                                    + (1 - alpha) * agg_vec_or_model[key])
            else:
                final_state[key] = own_vec_or_model[key]
        return final_state

    @abstractmethod
    def aggregate(self, own_vec, neighbor_vecs, **kwargs):
        """
        聚合邻居向量。
        own_vec: Tensor [D]
        neighbor_vecs: List[Tensor [D]] 或 Tensor [N, D]
        返回: Tensor [D]
        """
        pass

    @staticmethod
    def compute_cosine_similarity(vec1, vec2, eps=1e-8):
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        if norm1 < eps or norm2 < eps:
            return 0.0
        return (torch.dot(vec1, vec2) / (norm1 * norm2)).item()
