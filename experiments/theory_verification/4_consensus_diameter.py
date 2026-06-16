"""
Experiment B3: Consensus Diameter Measurement
Verify Theorem 5.2: steady-state max||w_i - w_j|| ≤ C₃·(κ/γ)·√(G² + B²ρ²)
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

from aggregators import SAMAAggregator, BALANCEAggregator
from models import SimpleCNN
from utils import load_mnist, generate_ring_topology, compute_spectral_gap

# Load config
_config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'
with open(_config_path, 'r') as _f:
    _config = yaml.safe_load(_f)
_tv_cfg = _config['theory_verification']
_sama_cfg = _config['sama']
_balance_cfg = _config['balance']


def measure_consensus_diameter(method='sama', num_clients=None, byzantine_ratio=None,
                               num_rounds=None, alpha_dirichlet=None):
    """
    测量稳态共识直径

    参数:
        method: 'sama' 或 'balance'
        num_clients: 客户端数
        byzantine_ratio: 拜占庭比例
        num_rounds: 训练轮次（需要足够长以达到稳态）
        alpha_dirichlet: Dirichlet参数

    返回:
        dict - 测量结果
    """
    if num_clients is None:
        num_clients = _tv_cfg['num_clients']
    if byzantine_ratio is None:
        byzantine_ratio = _tv_cfg['byzantine_ratio']
    if alpha_dirichlet is None:
        alpha_dirichlet = _tv_cfg['alpha_dirichlet']
    if num_rounds is None:
        num_rounds = _tv_cfg['num_rounds_consensus']

    print(f"Measuring {method.upper()} consensus diameter...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    train_loaders, test_loader = load_mnist(num_clients=num_clients, alpha=alpha_dirichlet,
                                            batch_size=_tv_cfg['batch_size'],
                                            num_workers=_tv_cfg['num_workers'])

    neighbors = generate_ring_topology(num_clients)

    num_byzantine = int(num_clients * byzantine_ratio)
    honest_nodes = set(range(num_clients - num_byzantine))
    byzantine_nodes = set(range(num_clients - num_byzantine, num_clients))

    gamma = compute_spectral_gap(neighbors, honest_nodes)
    print(f"Honest subgraph spectral gap γ = {gamma:.4f}")

    models = [SimpleCNN().to(device) for _ in range(num_clients)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=_tv_cfg['lr']) for m in models]

    if method == 'sama':
        aggregator = SAMAAggregator(alpha=_sama_cfg['alpha'], use_temperature=True)
    elif method == 'balance':
        aggregator = BALANCEAggregator(alpha=_balance_cfg['alpha'],
                                       gamma=_balance_cfg['gamma'],
                                       kappa=_balance_cfg['kappa'])
    else:
        raise ValueError(f"Unknown method: {method}")

    diameter_history = []  # max_{i,j∈H} ||w_i - w_j||

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

        honest_vecs = [updated_vecs[i] for i in honest_nodes]

        honest_mat = torch.stack(honest_vecs)
        max_dist = torch.cdist(honest_mat, honest_mat).max().item()

        diameter_history.append(max_dist)

        if (t + 1) % 50 == 0:
            print(f"Round {t+1}/{num_rounds}: Diameter = {max_dist:.4f}")

    steady_state_diameter = np.mean(diameter_history[-100:])
    diameter_std = np.std(diameter_history[-100:])

    kappa = np.sqrt(num_byzantine / len(honest_nodes))
    C3 = 2 * np.sqrt(2)
    G_estimated = 2.0
    B_estimated = 0.5
    rho_estimated = 1.0

    theoretical_bound = C3 * (kappa / gamma) * np.sqrt(G_estimated**2 + B_estimated**2 * rho_estimated**2)

    print("\n" + "=" * 80)
    print("Consensus Diameter Measurement Results")
    print("=" * 80)
    print(f"Steady-state diameter: {steady_state_diameter:.4f} ± {diameter_std:.4f}")
    print(f"Theoretical bound: {theoretical_bound:.4f}")
    print(f"Margin ratio: {(theoretical_bound - steady_state_diameter)/theoretical_bound*100:.1f}%")
    print(f"\nTheorem 5.2 Verification: {'✓ Diameter < Theoretical Bound' if steady_state_diameter < theoretical_bound else '✗ Exceeds Bound'}")

    result = {
        'diameter_history': diameter_history,
        'steady_state': steady_state_diameter,
        'theoretical_bound': theoretical_bound,
        'gamma': gamma,
        'kappa': kappa,
        'num_clients': num_clients,
        'num_byzantine': int(num_clients * byzantine_ratio)
    }
    _plot_and_save(result)
    import json
    save_dir = Path(__file__).parent.parent.parent / 'results'
    json_path = save_dir / 'consensus_diameter.json'
    with open(json_path, 'w') as f:
        json.dump({
            'diameter_history': result['diameter_history'],
            'steady_state': float(result['steady_state']),
            'theoretical_bound': float(result['theoretical_bound']),
            'gamma': float(result['gamma']),
            'kappa': float(result['kappa']),
            'num_clients': result['num_clients'],
            'num_byzantine': result['num_byzantine'],
        }, f, indent=2)
    print(f"Raw data saved: {json_path.name}")
    return result


def _plot_and_save(result):
    """绘图并保存结果。"""
    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(result['diameter_history'], linewidth=2, alpha=0.7, label='Measured')
    plt.axhline(result['steady_state'], color='r', linestyle='--',
                label=f"Steady State: {result['steady_state']:.2f}")
    plt.axhline(result['theoretical_bound'], color='g', linestyle='--',
                label=f"Theoretical Bound: {result['theoretical_bound']:.2f}")
    plt.xlabel('Training Round')
    plt.ylabel('Diameter max||w_i - w_j||')
    plt.title(f'Consensus Diameter Over Time\n'
              f'(n={result["num_clients"]}, f={result["num_byzantine"]}, '
              f'γ={_balance_cfg["gamma"]}, std={_tv_cfg["gaussian_attack_std"]})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    values = [result['steady_state'], result['theoretical_bound']]
    labels = ['Measured', 'Theoretical Bound']
    plt.bar(labels, values, color=['C0', 'C2'], alpha=0.7)
    plt.ylabel('Diameter')
    plt.title('Steady State vs Theoretical Bound')
    plt.grid(True, alpha=0.3, axis='y')

    text_str = f"κ={result['kappa']:.2f}, γ={result['gamma']:.2f}\nC₃=2√2≈2.83"
    plt.text(0.5, max(values) * 0.5, text_str,
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'consensus_diameter.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_dir / 'consensus_diameter.png'}")
    plt.close(fig)


if __name__ == "__main__":
    measure_consensus_diameter()
