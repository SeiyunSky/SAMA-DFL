"""
Byzantine attack implementations
"""
import torch
import numpy as np
from collections import OrderedDict


class GaussianAttack:
    """
    高斯攻击: 用随机高斯噪声替换梯度
    g_mal ~ N(0, σ²)
    """
    def __init__(self, std=10.0):
        self.std = std
        self.name = "Gaussian"

    def attack(self, model_state):
        """
        生成恶意模型

        参数:
            model_state: OrderedDict - 正常的模型参数

        返回:
            OrderedDict - 被攻击后的模型
        """
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
        """
        翻转标签

        参数:
            labels: Tensor - 原始标签

        返回:
            Tensor - 翻转后的标签
        """
        return (self.num_classes - 1) - labels

    def attack_dataset(self, dataset):
        """
        返回标签翻转后的数据集
        （需要在数据加载阶段调用）
        """
        # 这个方法在Client类中实现
        pass


class OmniscientAttack:
    """
    全知攻击: 利用已知的诚实梯度之和，发送其负值
    g_mal = -C · Σ g_honest
    """
    def __init__(self, amplification=2.0):
        self.amplification = amplification
        self.name = "Omniscient"

    def attack(self, honest_models):
        """
        生成针对性的恶意模型

        参数:
            honest_models: List[OrderedDict] - 诚实节点的模型

        返回:
            OrderedDict - 恶意模型
        """
        if not honest_models:
            return honest_models[0]  # 降级为不攻击

        # 计算诚实模型的平均
        avg_state = OrderedDict()
        for key in honest_models[0].keys():
            if isinstance(honest_models[0][key], torch.Tensor):
                stacked = torch.stack([m[key] for m in honest_models])
                avg_state[key] = stacked.mean(dim=0)
            else:
                avg_state[key] = honest_models[0][key]

        # 生成反向攻击
        malicious_state = OrderedDict()
        for key, param in avg_state.items():
            if isinstance(param, torch.Tensor):
                malicious_state[key] = -self.amplification * param
            else:
                malicious_state[key] = param

        return malicious_state


# 攻击工厂
ATTACK_REGISTRY = {
    'gaussian': GaussianAttack,
    'label_flipping': LabelFlippingAttack,
    'omniscient': OmniscientAttack,
}


def get_attack(attack_name, **kwargs):
    """获取攻击实例"""
    if attack_name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: {attack_name}")
    return ATTACK_REGISTRY[attack_name](**kwargs)
