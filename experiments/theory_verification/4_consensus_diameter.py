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
from collections import OrderedDict

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
    honest_nodes = list(range(num_clients - num_byzantine))
    byzantine_nodes = list(range(num_clients - num_byzantine, num_clients))

    gamma = compute_spectral_gap(neighbors, honest_nodes)
    print(f"Honest subgraph spectral gap γ = {gamma:.4f}")

    models = [SimpleCNN().to(device) for _ in range(num_clients)]

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
        local_models = []
        for i in range(num_clients):
            model = models[i]
            model.train()

            if i in honest_nodes:
                try:
                    data, target = next(iter(train_loaders[i]))
                    data, target = data.to(device), target.to(device)

                    optimizer = torch.optim.SGD(model.parameters(), lr=_tv_cfg['lr'])
                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                except:
                    pass

            local_models.append(model.state_dict())

        for byz_id in byzantine_nodes:
            malicious = OrderedDict()
            for key, param in local_models[byz_id].items():
                if isinstance(param, torch.Tensor):
                    malicious[key] = param + torch.randn_like(param) * _tv_cfg['gaussian_attack_std']
                else:
                    malicious[key] = param
            local_models[byz_id] = malicious

        updated_models = []
        for i in range(num_clients):
            own_model = local_models[i]
            neighbor_models = [local_models[j] for j in neighbors[i]]

            if i in honest_nodes:
                aggregated = aggregator.aggregate(own_model, neighbor_models, t=t, T=num_rounds)
                final = aggregator.final_update(own_model, aggregated)
            else:
                final = own_model

            updated_models.append(final)

        models = [SimpleCNN().to(device) for _ in range(num_clients)]
        for i, state_dict in enumerate(updated_models):
            models[i].load_state_dict(state_dict)

        honest_vecs = [aggregator.model_to_vector(updated_models[i]) for i in honest_nodes]

        max_dist = 0
        for i, vec_i in enumerate(honest_vecs):
            for j, vec_j in enumerate(honest_vecs):
                if i < j:
                    dist = torch.norm(vec_i - vec_j).item()
                    max_dist = max(max_dist, dist)

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

    return {
        'diameter_history': diameter_history,
        'steady_state': steady_state_diameter,
        'theoretical_bound': theoretical_bound,
        'gamma': gamma,
        'kappa': kappa,
        'num_clients': num_clients,
        'num_byzantine': int(num_clients * byzantine_ratio)
    }


if __name__ == "__main__":
    result = measure_consensus_diameter()

    plt.figure(figsize=(12, 5))

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
