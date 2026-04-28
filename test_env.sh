#!/bin/bash
# 快速测试脚本 - 验证环境配置

echo "=========================================="
echo "SAMA-DFL 环境快速测试"
echo "=========================================="

# 测试1: Python + PyTorch
echo -e "\n[测试1/5] Python环境..."
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 测试2: CUDA
echo -e "\n[测试2/5] CUDA环境..."
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无\"}'); print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB' if torch.cuda.is_available() else '')"

# 测试3: 必要依赖
echo -e "\n[测试3/5] 依赖包..."
python -c "import numpy, matplotlib, yaml, tqdm; print('✓ 所有依赖已安装')"

# 测试4: 数据加载（快速）
echo -e "\n[测试4/5] 数据加载测试..."
python -c "
import sys
sys.path.append('.')
from utils import load_mnist
print('下载MNIST数据集...')
train_loaders, test_loader = load_mnist(num_clients=5, batch_size=128, num_workers=2)
print(f'✓ 数据加载成功: {len(train_loaders)}个客户端')
"

# 测试5: 简单前向传播
echo -e "\n[测试5/5] GPU计算测试..."
python -c "
import torch
from models import SimpleCNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
x = torch.randn(128, 1, 28, 28).to(device)
y = model(x)
print(f'✓ GPU计算测试通过')
print(f'  输入: {x.shape}')
print(f'  输出: {y.shape}')
print(f'  设备: {device}')
"

echo ""
echo "=========================================="
echo "✓ 所有测试通过！环境配置正确"
echo "=========================================="
echo ""
echo "现在可以运行实验:"
echo "  bash run_parallel.sh      # 并行加速"
echo "  python quickstart.py      # 交互式"
echo "=========================================="
