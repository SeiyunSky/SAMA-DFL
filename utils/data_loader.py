"""
Data loading utilities with Non-IID partitioning
"""
import numpy as np
import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


# ──────────────────────────────────────────────────────────
# GPU 预加载 DataLoader（避免 CPU→GPU 拷贝瓶颈）
# ──────────────────────────────────────────────────────────

class GPUTensorLoader:
    """
    把整个数据子集预先放到 GPU，迭代时直接切片返回。
    相比标准 DataLoader 完全消除 host→device 拷贝和 worker 进程开销。

    注意：MNIST 全量 ~47MB、CIFAR-10 全量 ~150MB，单卡 GPU 显存可轻松容纳。
    """
    def __init__(self, data, targets, batch_size, shuffle=True, device='cuda'):
        # data: Tensor[N, ...], targets: Tensor[N]
        self.data = data.to(device, non_blocking=True)
        self.targets = targets.to(device, non_blocking=True)
        self.batch_size = min(batch_size, len(self.data))
        self.shuffle = shuffle
        self.device = device

    def __len__(self):
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.data)
        if self.shuffle:
            perm = torch.randperm(n, device=self.device)
        else:
            perm = torch.arange(n, device=self.device)
        for start in range(0, n, self.batch_size):
            idx = perm[start:start + self.batch_size]
            yield self.data[idx], self.targets[idx]


# ──────────────────────────────────────────────────────────
# 数据集检测与下载
# ──────────────────────────────────────────────────────────

# 每个数据集在 data_dir 下预期存在的标志文件/目录
_DATASET_MARKERS = {
    'mnist': [
        'MNIST/raw/train-images-idx3-ubyte',
        'MNIST/raw/t10k-images-idx3-ubyte',
    ],
    'cifar10': [
        'cifar-10-batches-py/data_batch_1',
        'cifar-10-batches-py/test_batch',
    ],
}


def check_dataset(name: str, data_dir: str = './data') -> bool:
    """检查数据集是否已下载完整，返回 True/False"""
    root = Path(data_dir)
    markers = _DATASET_MARKERS.get(name.lower(), [])
    return all((root / m).exists() for m in markers)


def check_all_datasets(data_dir: str = './data') -> dict:
    """返回 {dataset_name: bool} 的完整状态字典"""
    return {name: check_dataset(name, data_dir) for name in _DATASET_MARKERS}


def download_dataset(name: str, data_dir: str = './data'):
    """下载指定数据集（仅下载，不做划分）"""
    name = name.lower()
    transform = transforms.ToTensor()
    if name == 'mnist':
        datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    elif name == 'cifar10':
        datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {name}")



def dirichlet_partition(dataset, num_clients, alpha=0.1, num_classes=10):
    """
    使用Dirichlet分布划分数据集（Non-IID）

    参数:
        dataset: torchvision.datasets - 数据集
        num_clients: int - 客户端数量
        alpha: float - Dirichlet参数（越小越异质）
        num_classes: int - 类别数

    返回:
        List[List[int]] - 每个客户端的数据索引
    """
    # 按类别组织数据索引
    labels = np.array(dataset.targets)
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    # 为每个客户端分配数据
    client_indices = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        # 生成Dirichlet分布
        proportions = np.random.dirichlet([alpha] * num_clients)

        # 根据比例分配数据
        split_points = (np.cumsum(proportions) * len(c_idx)).astype(int)[:-1]
        splits = np.split(c_idx, split_points)

        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    # 打乱每个客户端的数据
    for indices in client_indices:
        np.random.shuffle(indices)

    return client_indices


def load_mnist(data_dir='./data', num_clients=20, alpha=0.1, batch_size=32, num_workers=4):
    """
    加载MNIST数据集并进行Non-IID划分

    参数:
        num_workers: 数据加载工作进程数（默认4，vGPU-32GB可用8）

    返回:
        train_loaders: List[DataLoader] - 每个客户端的训练数据加载器
        test_loader: DataLoader - 全局测试数据加载器
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=transform)

    # Non-IID划分
    client_indices = dirichlet_partition(train_dataset, num_clients, alpha=alpha, num_classes=10)

    # 创建每个客户端的DataLoader
    train_loaders = []
    for i, indices in enumerate(client_indices):
        # 检查是否为空，如果为空则分配至少1个样本
        if len(indices) == 0:
            print(f"Warning: Client {i} has 0 samples, assigning 1 sample")
            indices = [0]  # 分配第一个样本

        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=min(batch_size, len(indices)),  # batch_size不能超过样本数
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        train_loaders.append(loader)

    # 全局测试集
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loaders, test_loader


def load_cifar10(data_dir='./data', num_clients=20, alpha=0.1, batch_size=32, num_workers=4):
    """
    加载CIFAR-10数据集并进行Non-IID划分

    参数:
        num_workers: 数据加载工作进程数
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform_test)

    # Non-IID划分
    client_indices = dirichlet_partition(train_dataset, num_clients, alpha=alpha, num_classes=10)

    # 创建DataLoader
    train_loaders = []
    for i, indices in enumerate(client_indices):
        # 检查是否为空，如果为空则分配至少1个样本
        if len(indices) == 0:
            print(f"Warning: Client {i} has 0 samples, assigning 1 sample")
            indices = [0]  # 分配第一个样本

        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=min(batch_size, len(indices)),  # batch_size不能超过样本数
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
        train_loaders.append(loader)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loaders, test_loader


# ──────────────────────────────────────────────────────────
# GPU 预加载版（推荐，比 CPU DataLoader 快 5-10x）
# ──────────────────────────────────────────────────────────

def _materialize_dataset_to_tensors(dataset, normalize_mean, normalize_std):
    """把 torchvision dataset 全量转换为 (data_tensor, target_tensor)，已归一化。
    避开 dataset.__getitem__ 的 transform 开销（一次性 vectorize）。
    """
    # MNIST: dataset.data 是 uint8 [N, 28, 28]
    # CIFAR10: dataset.data 是 uint8 [N, 32, 32, 3] (numpy)
    raw = dataset.data
    if isinstance(raw, np.ndarray):
        raw = torch.from_numpy(raw)
    raw = raw.float() / 255.0

    # 标准化形状到 [N, C, H, W]
    if raw.dim() == 3:  # MNIST
        raw = raw.unsqueeze(1)  # [N, 1, H, W]
    elif raw.dim() == 4 and raw.shape[-1] in (1, 3):  # CIFAR10 NHWC
        raw = raw.permute(0, 3, 1, 2).contiguous()

    # 归一化（per-channel）
    mean = torch.tensor(normalize_mean).view(1, -1, 1, 1)
    std = torch.tensor(normalize_std).view(1, -1, 1, 1)
    raw = (raw - mean) / std

    targets = dataset.targets
    if isinstance(targets, list):
        targets = torch.tensor(targets, dtype=torch.long)
    elif isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).long()
    elif isinstance(targets, torch.Tensor):
        targets = targets.long()

    return raw, targets


def load_mnist_gpu(data_dir='./data', num_clients=20, alpha=0.1, batch_size=128, device='cuda'):
    """MNIST GPU 预加载版：全数据放显存，DataLoader 是 GPU tensor 切片迭代器。"""
    train_dataset = datasets.MNIST(data_dir, train=True, download=False, transform=None)
    test_dataset = datasets.MNIST(data_dir, train=False, download=False, transform=None)

    train_data, train_targets = _materialize_dataset_to_tensors(
        train_dataset, [0.1307], [0.3081])
    test_data, test_targets = _materialize_dataset_to_tensors(
        test_dataset, [0.1307], [0.3081])

    # Non-IID 划分
    client_indices = dirichlet_partition(train_dataset, num_clients, alpha=alpha, num_classes=10)

    train_loaders = []
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            indices = [0]
        idx_t = torch.tensor(indices, dtype=torch.long)
        loader = GPUTensorLoader(
            train_data[idx_t], train_targets[idx_t],
            batch_size=batch_size, shuffle=True, device=device,
        )
        train_loaders.append(loader)

    test_loader = GPUTensorLoader(
        test_data, test_targets,
        batch_size=batch_size * 2, shuffle=False, device=device,
    )
    return train_loaders, test_loader


def load_cifar10_gpu(data_dir='./data', num_clients=20, alpha=0.1, batch_size=128, device='cuda'):
    """CIFAR-10 GPU 预加载版（无 augmentation——为了能预 load 到 GPU，
    数据增强需要每 batch 重做，无法预 load）。
    """
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=None)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=None)

    train_data, train_targets = _materialize_dataset_to_tensors(
        train_dataset,
        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    test_data, test_targets = _materialize_dataset_to_tensors(
        test_dataset,
        [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

    client_indices = dirichlet_partition(train_dataset, num_clients, alpha=alpha, num_classes=10)

    train_loaders = []
    for i, indices in enumerate(client_indices):
        if len(indices) == 0:
            indices = [0]
        idx_t = torch.tensor(indices, dtype=torch.long)
        loader = GPUTensorLoader(
            train_data[idx_t], train_targets[idx_t],
            batch_size=batch_size, shuffle=True, device=device,
        )
        train_loaders.append(loader)

    test_loader = GPUTensorLoader(
        test_data, test_targets,
        batch_size=batch_size * 2, shuffle=False, device=device,
    )
    return train_loaders, test_loader


if __name__ == "__main__":
    # 测试
    print("Testing data loading with vGPU-32GB optimization...")
    train_loaders, test_loader = load_mnist(num_clients=10, alpha=0.1, batch_size=128, num_workers=8)
    print(f"Created {len(train_loaders)} client loaders")
    print(f"Batch size: 128 (optimized for vGPU-32GB)")
    print(f"Num workers: 8 (utilizing 12-core CPU)")
    print(f"Test loader batches: {len(test_loader)}")

    # 检查数据分布
    for i, loader in enumerate(train_loaders[:3]):
        labels = []
        for _, y in loader:
            labels.extend(y.numpy())
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Client {i}: {dict(zip(unique, counts))}")
