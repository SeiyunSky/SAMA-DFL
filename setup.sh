#!/bin/bash
# AutoDL快速启动脚本 - vGPU-32GB优化版

echo "=========================================="
echo "SAMA-DFL Experiments - AutoDL Setup"
echo "配置: vGPU-32GB, 12核CPU, 90GB内存"
echo "=========================================="

# 1. 检查CUDA
echo -e "\n[1/5] 检查CUDA环境..."
nvidia-smi
echo ""

# 2. 安装依赖
echo -e "\n[2/5] 安装Python依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 创建结果目录
echo -e "\n[3/5] 创建工作目录..."
mkdir -p results data logs

# 4. 验证安装
echo -e "\n[4/5] 验证PyTorch + CUDA..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 5. 显示配置优化信息
echo -e "\n[5/5] 配置优化已应用:"
echo "  ✓ batch_size: 32 → 128 (利用32GB显存)"
echo "  ✓ learning_rate: 0.01 → 0.02 (适配大batch)"
echo "  ✓ 并行执行脚本已准备: run_parallel.sh"

# 准备就绪
echo -e "\n=========================================="
echo "准备就绪！推荐运行方式:"
echo "=========================================="
echo ""
echo "🚀 方式1: 并行加速 (推荐, 1.5小时, 2.5元)"
echo "  bash run_parallel.sh"
echo ""
echo "📝 方式2: 交互式选择"
echo "  python quickstart.py"
echo ""
echo "⚙️  方式3: 命令行精确控制"
echo "  python run_experiments.py --mode theory       # 理论验证"
echo "  python run_experiments.py --mode performance  # 性能实验"
echo "  python run_experiments.py --experiment mnist  # 单个实验"
echo ""
echo "=========================================="
echo "💡 提示: 并行运行可节省50%时间！"
echo "=========================================="
