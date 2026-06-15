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

    def final_update(self, local_model, aggregated_model, alpha=None, **kwargs):
        """
        自锚定融合: w_i^{t+1} = α·w_i' + (1-α)·AGG_i
        """
        if alpha is None:
            alpha = self.alpha
        final_state = OrderedDict()
        for key in local_model.keys():
            if isinstance(local_model[key], torch.Tensor):
                final_state[key] = alpha * local_model[key] + (1 - alpha) * aggregated_model[key]
            else:
                final_state[key] = local_model[key]
        return final_state

    @abstractmethod
    def aggregate(self, own_model, neighbor_models, **kwargs):
        """
        聚合邻居模型

        参数:
            own_model: OrderedDict - 自身模型的state_dict
            neighbor_models: List[OrderedDict] - 邻居模型列表
            **kwargs: 其他参数

        返回:
            OrderedDict - 聚合后的模型
        """
        pass

    @staticmethod
    def model_to_vector(state_dict):
        """将模型参数扁平化为向量"""
        vectors = []
        for key in sorted(state_dict.keys()):
            param = state_dict[key]
            if isinstance(param, torch.Tensor):
                vectors.append(param.reshape(-1).float())
        return torch.cat(vectors)

    @staticmethod
    def vector_to_model(vector, reference_state_dict):
        """将向量恢复为模型参数字典"""
        new_state = OrderedDict()
        idx = 0
        for key in sorted(reference_state_dict.keys()):
            param = reference_state_dict[key]
            if isinstance(param, torch.Tensor):
                param_shape = param.shape
                param_numel = param.numel()
                new_state[key] = vector[idx:idx+param_numel].reshape(param_shape)
                idx += param_numel
            else:
                new_state[key] = param  # 保留非Tensor项
        return new_state

    @staticmethod
    def compute_cosine_similarity(vec1, vec2, eps=1e-8):
        """计算两个向量的余弦相似度"""
        norm1 = torch.norm(vec1)
        norm2 = torch.norm(vec2)
        if norm1 < eps or norm2 < eps:
            return 0.0
        cos_sim = torch.dot(vec1, vec2) / (norm1 * norm2)
        return cos_sim.item()
