"""
Neural network models for experiments
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    简单CNN模型（用于MNIST）

    架构:
    - Conv1: 1 → 32 channels, 5x5 kernel
    - Conv2: 32 → 64 channels, 5x5 kernel
    - FC1: 64*4*4 → 512
    - FC2: 512 → 10
    """

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28x28 → 14x14
        x = self.pool(F.relu(self.conv2(x)))  # 14x14 → 7x7
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TwoLayerMLP(nn.Module):
    """
    简单的两层MLP（用于合成数据和快速测试）
    """
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 模型注册表
MODEL_REGISTRY = {
    'simple_cnn': SimpleCNN,
    'mlp': TwoLayerMLP,
}


def get_model(model_name, **kwargs):
    """获取模型实例"""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)
