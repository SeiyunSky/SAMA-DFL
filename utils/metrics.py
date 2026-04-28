"""
Evaluation metrics for federated learning
"""
import torch
import numpy as np


def compute_accuracy(model, data_loader, device):
    """
    计算模型准确率

    参数:
        model: torch.nn.Module
        data_loader: DataLoader
        device: torch device

    返回:
        float - 准确率 (0-100)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


def compute_loss(model, data_loader, device, criterion=None):
    """
    计算模型平均损失

    参数:
        model: torch.nn.Module
        data_loader: DataLoader
        device: torch device
        criterion: loss function (default: CrossEntropy)

    返回:
        float - 平均损失
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    loss_sum = 0
    total = 0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_sum += criterion(output, target).item() * target.size(0)
            total += target.size(0)

    return loss_sum / total


def compute_consensus_error(models, honest_nodes, aggregator):
    """
    计算共识误差 D = (1/|H|)Σ||w_i - w̄_H||²

    参数:
        models: List[state_dict] - 模型列表
        honest_nodes: List[int] - 诚实节点索引
        aggregator: BaseAggregator - 用于向量转换

    返回:
        float - 共识误差
    """
    honest_vecs = []
    for i in honest_nodes:
        vec = aggregator.model_to_vector(models[i])
        honest_vecs.append(vec)

    honest_vecs = torch.stack(honest_vecs)
    honest_mean = honest_vecs.mean(dim=0)

    D = torch.mean(torch.norm(honest_vecs - honest_mean, dim=1).pow(2)).item()
    return D


def compute_model_divergence(models, honest_nodes, aggregator):
    """
    计算模型散度 (最大成对距离)

    参数:
        models: List[state_dict]
        honest_nodes: List[int]
        aggregator: BaseAggregator

    返回:
        float - 最大成对距离
    """
    honest_vecs = []
    for i in honest_nodes:
        vec = aggregator.model_to_vector(models[i])
        honest_vecs.append(vec)

    max_dist = 0
    for i in range(len(honest_vecs)):
        for j in range(i + 1, len(honest_vecs)):
            dist = torch.norm(honest_vecs[i] - honest_vecs[j]).item()
            max_dist = max(max_dist, dist)

    return max_dist


def compute_gradient_norm(model, data_loader, device):
    """
    计算批次梯度范数

    参数:
        model: torch.nn.Module
        data_loader: DataLoader
        device: torch device

    返回:
        float - 梯度L2范数
    """
    model.train()

    try:
        data, target = next(iter(data_loader))
        data, target = data.to(device), target.to(device)

        model.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()

        grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2

        return np.sqrt(grad_norm)
    except:
        return 0.0


def evaluate_all_metrics(model, test_loader, device):
    """
    一次性计算所有评估指标

    返回:
        dict - {'accuracy', 'loss', 'top5_acc'}
    """
    model.eval()
    correct = 0
    top5_correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # Loss
            loss_sum += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()

            # Top-1 accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            # Top-5 accuracy
            _, top5_pred = output.topk(5, dim=1)
            top5_correct += top5_pred.eq(target.view(-1, 1).expand_as(top5_pred)).sum().item()

            total += target.size(0)

    return {
        'accuracy': 100.0 * correct / total,
        'loss': loss_sum / total,
        'top5_acc': 100.0 * top5_correct / total
    }


__all__ = [
    'compute_accuracy',
    'compute_loss',
    'compute_consensus_error',
    'compute_model_divergence',
    'compute_gradient_norm',
    'evaluate_all_metrics'
]
