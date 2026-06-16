"""
Experiment B4: Lyapunov Function Verification
Verify Lyapunov function properties in Theorems 5.1 and 5.2
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import SAMAAggregator
from models import SimpleCNN
from utils import load_mnist, generate_ring_topology

# Load config
_config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'
with open(_config_path, 'r') as _f:
    _config = yaml.safe_load(_f)
_tv_cfg = _config['theory_verification']
_sama_cfg = _config['sama']


def compute_lyapunov_function(honest_vecs_list, honest_nodes, optimal_loss=0.0):
    """
    计算Lyapunov函数 V_t = ε_t + ρ·D_t

    其中:
        ε_t = F(w̄_H) - F* (优化误差)
        D_t = (1/|H|)Σ||w_i - w̄_H||² (共识误差)

    参数:
        honest_vecs_list: List[Tensor] - 诚实节点的模型向量
        honest_nodes: List[int] - 诚实节点索引（用于长度信息，不直接索引）
        optimal_loss: float - 最优损失F*（近似）

    返回:
        tuple - (D_t, honest_mean)
    """
    honest_vecs = torch.stack(honest_vecs_list)
    honest_mean = honest_vecs.mean(dim=0)

    # 共识误差 D_t
    D_t = torch.mean(torch.norm(honest_vecs - honest_mean, dim=1).pow(2)).item()

    return D_t, honest_mean


def run_lyapunov_verification(num_clients=None, byzantine_ratio=None, num_rounds=None):
    """
    运行Lyapunov函数验证实验

    验证点:
    1. V_t单调递减（大部分时候）
    2. 收敛到稳态后V_t趋于常数
    """
    if num_clients is None:
        num_clients = _tv_cfg['num_clients']
    if byzantine_ratio is None:
        byzantine_ratio = _tv_cfg['byzantine_ratio']
    if num_rounds is None:
        num_rounds = _tv_cfg['num_rounds_lyapunov']
    print("=" * 80)
    print("Experiment B4: Lyapunov Function Verification")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading with num_workers=0
    train_loaders, test_loader = load_mnist(num_clients=num_clients, alpha=_tv_cfg['alpha_dirichlet'],
                                            batch_size=_tv_cfg['batch_size'],
                                            num_workers=_tv_cfg['num_workers'])

    # 网络拓扑
    neighbors = generate_ring_topology(num_clients)

    # 节点划分
    num_byzantine = int(num_clients * byzantine_ratio)
    honest_nodes = set(range(num_clients - num_byzantine))
    byzantine_nodes = set(range(num_clients - num_byzantine, num_clients))

    models = [SimpleCNN().to(device) for _ in range(num_clients)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=_tv_cfg['lr']) for m in models]
    aggregator = SAMAAggregator(alpha=_sama_cfg['alpha'], use_temperature=True)
    global_model = SimpleCNN().to(device)

    D_history = []
    loss_history = []
    V_history = []
    delta_V_history = []

    for t in range(num_rounds):
        local_vecs = [None] * num_clients
        for i in range(num_clients):
            model = models[i]
            model.train()

            if i in honest_nodes:
                try:
                    data, target = next(iter(train_loaders[i]))
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                    optimizer = optimizers[i]
                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                except:
                    pass

            local_vecs[i] = aggregator.model_to_vector(models[i])

        for byz_id in byzantine_nodes:
            local_vecs[byz_id] = local_vecs[byz_id] + torch.randn_like(local_vecs[byz_id]) * _tv_cfg['gaussian_attack_std']

        all_vecs = torch.stack(local_vecs)
        updated_vecs = [None] * num_clients
        for i in range(num_clients):
            own_vec = local_vecs[i]
            neighbor_vecs = all_vecs[neighbors[i]]

            if i in honest_nodes:
                aggregated = aggregator.aggregate(own_vec, neighbor_vecs, t=t, T=num_rounds)
                final_vec = aggregator.final_update(own_vec, aggregated)
            else:
                final_vec = own_vec

            updated_vecs[i] = final_vec

        for i, vec in enumerate(updated_vecs):
            aggregator.load_from_vector(models[i], vec)

        # 计算Lyapunov函数
        honest_vecs_t = [updated_vecs[i] for i in honest_nodes]
        D_t, honest_mean = compute_lyapunov_function(honest_vecs_t, honest_nodes)

        # 评估诚实平均模型的损失
        aggregator.load_from_vector(global_model, honest_mean)
        global_model.eval()

        total_loss = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                output = global_model(data)
                total_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
                total += target.size(0)

        avg_loss = total_loss / total

        rho = _tv_cfg['lyapunov_rho']
        V_t = avg_loss + rho * D_t

        D_history.append(D_t)
        loss_history.append(avg_loss)
        V_history.append(V_t)

        if len(V_history) > 1:
            delta_V = V_history[-1] - V_history[-2]
            delta_V_history.append(delta_V)

        if (t + 1) % 50 == 0:
            print(f"Round {t+1}/{num_rounds}: V_t={V_t:.4f}, Loss={avg_loss:.4f}, D_t={D_t:.4f}")

    # 分析
    delta_V_arr = np.array(delta_V_history)
    num_decrease = np.sum(delta_V_arr < 0)
    decrease_ratio = num_decrease / len(delta_V_arr) * 100

    steady_state_V = np.mean(V_history[-100:])
    steady_state_std = np.std(V_history[-100:])

    print("\n" + "=" * 80)
    print("Lyapunov Function Verification Results")
    print("=" * 80)
    print(f"Proportion of ΔV < 0: {decrease_ratio:.1f}%")
    print(f"Steady-state V∞: {steady_state_V:.4f} ± {steady_state_std:.4f}")
    print(f"Steady-state relative variation: {steady_state_std/steady_state_V*100:.2f}%")

    # Judgment
    is_mostly_decreasing = decrease_ratio > 70
    is_converged = steady_state_std / steady_state_V < 0.05

    print(f"\n✓ Monotonicity verification: {'PASS' if is_mostly_decreasing else 'FAIL'} (>70% decreasing)")
    print(f"✓ Convergence verification: {'PASS' if is_converged else 'FAIL'} (<5% variation)")

    result = {
        'V_history': V_history,
        'D_history': D_history,
        'loss_history': loss_history,
        'delta_V_history': delta_V_history,
        'decrease_ratio': decrease_ratio,
        'steady_state_V': steady_state_V
    }

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(result['V_history'], linewidth=2, alpha=0.7)
    axes[0, 0].axhline(result['steady_state_V'], color='r', linestyle='--',
                      label=f"Steady State: {result['steady_state_V']:.2f}")
    axes[0, 0].set_xlabel('Training Round')
    axes[0, 0].set_ylabel('Lyapunov Function $V_t$')
    axes[0, 0].set_title(f'Lyapunov Function Over Time\n'
                         f'(n={_tv_cfg["num_clients"]}, f={int(_tv_cfg["num_clients"]*_tv_cfg["byzantine_ratio"])}, '
                         f'lr={_tv_cfg["lr"]}, std={_tv_cfg["gaussian_attack_std"]}, α={_sama_cfg["alpha"]})')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].hist(result['delta_V_history'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(0, color='r', linestyle='--', linewidth=2, label='ΔV=0')
    axes[0, 1].set_xlabel('ΔV_t = V_t - V_{t-1}')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'ΔV Distribution ({result["decrease_ratio"]:.1f}% < 0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    ax1 = axes[1, 0]
    ax1.plot(result['loss_history'], color='C0', linewidth=2, label='Test Loss')
    ax1.set_xlabel('Training Round (×10)')
    ax1.set_ylabel('Test Loss', color='C0')
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(result['D_history'], color='C1', linewidth=2, label='Consensus Error')
    ax2.set_ylabel('Consensus Error $D_t$', color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    axes[1, 0].set_title('Loss and Consensus Error Decomposition')

    window = 20
    V_smooth = np.convolve(result['V_history'], np.ones(window)/window, mode='valid')
    axes[1, 1].plot(result['V_history'], alpha=0.3, color='gray', label='Raw')
    axes[1, 1].plot(np.arange(window-1, len(result['V_history'])), V_smooth,
                   linewidth=2, label=f'{window}-round Moving Average')
    axes[1, 1].set_xlabel('Training Round')
    axes[1, 1].set_ylabel('Lyapunov Function $V_t$')
    axes[1, 1].set_title('Lyapunov Function (Smoothed)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'lyapunov_verification.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_dir / 'lyapunov_verification.png'}")
    plt.close(fig)

    return result


if __name__ == "__main__":
    run_lyapunov_verification()
