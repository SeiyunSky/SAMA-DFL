"""
Experiment B2: Kappa Value Measurement
Measure and compare robustness constants of SAMA-DFL and BALANCE
Verify Lemma 4.4: kappa_SAMA < kappa_BALANCE
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import yaml
sys.path.append(str(Path(__file__).parent.parent.parent))

from aggregators import SAMAAggregator, BALANCEAggregator
from models import SimpleCNN
from utils import load_mnist, generate_ring_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack
from collections import OrderedDict

# Set matplotlib to avoid Chinese font issues
plt.rcParams['font.family'] = 'DejaVu Sans'

# Load config
_config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'
with open(_config_path, 'r') as _f:
    _config = yaml.safe_load(_f)
_tv_cfg = _config['theory_verification']
_sama_cfg = _config['sama']
_balance_cfg = _config['balance']


class KappaEstimator:
    """κ值估计器"""

    @staticmethod
    def estimate_kappa(own_vec, neighbor_vecs, aggregated_vec, honest_mask, aggregator_type='sama'):
        """
        从单次聚合估计κ值

        基于定义: ||AGG - w̄_H||² ≤ κ² · (1/|H|)·Σ||w_j - w̄_H||²

        参数:
            own_vec: Tensor - 自身模型向量
            neighbor_vecs: List[Tensor] - 邻居模型向量
            aggregated_vec: Tensor - 聚合结果向量
            honest_mask: List[bool] - 标记哪些邻居是诚实的
            aggregator_type: str - 'sama' 或 'balance'

        返回:
            float - 估计的κ值
        """
        # 提取诚实邻居
        honest_neighbors = [neighbor_vecs[j] for j, is_honest in enumerate(honest_mask)
                           if is_honest]

        if len(honest_neighbors) == 0:
            return None

        # 幅度对齐（仅SAMA）：AGG和w_bar_H必须在同一空间
        if aggregator_type == 'sama':
            # SAMA的AGG是幅度对齐向量的加权均值
            # w_bar_H也用幅度对齐后的诚实邻居均值，保证口径一致
            w_i_norm = torch.norm(own_vec)
            ref_vecs = []
            for w_j in honest_neighbors:
                w_j_norm = torch.norm(w_j)
                if w_j_norm > 1e-8:
                    ref_vecs.append(w_i_norm * (w_j / w_j_norm))
            if not ref_vecs:
                return None
        else:
            # BALANCE的AGG是原始向量均值，w_bar_H用原始向量
            ref_vecs = honest_neighbors

        # 计算诚实均值
        w_bar_H = torch.stack(ref_vecs).mean(dim=0)

        # 左侧: ||AGG - w̄_H||²
        left = torch.norm(aggregated_vec - w_bar_H).pow(2).item()

        # 右侧: (1/|H|)·Σ||w_j - w̄_H||²
        variances = [torch.norm(w - w_bar_H).pow(2).item() for w in ref_vecs]
        right = np.mean(variances)

        if right < 1e-8:
            return None

        # 估计κ
        kappa = np.sqrt(left / right)
        return kappa


def run_kappa_measurement(num_clients=None, byzantine_ratio=None, num_rounds=None,
                          alpha_dirichlet=None, topology_type=None, attack_type=None):
    """验证引理4.4: κ_SAMA < κ_BALANCE"""
    if num_clients is None:
        num_clients = _tv_cfg['num_clients']
    if byzantine_ratio is None:
        byzantine_ratio = _tv_cfg['byzantine_ratio']
    if alpha_dirichlet is None:
        alpha_dirichlet = _tv_cfg['alpha_dirichlet']
    if topology_type is None:
        topology_type = _tv_cfg['topology']
    if num_rounds is None:
        num_rounds = _tv_cfg['num_rounds_kappa']
    if attack_type is None:
        attack_type = _config['attack']['type']
    print("=" * 80)
    print("Experiment B2: Kappa Value Measurement")
    print("=" * 80)
    print(f"Configuration: n={num_clients}, f={int(byzantine_ratio*num_clients)}, "
          f"Dirichlet α={alpha_dirichlet}, topology={topology_type}, attack={attack_type}")
    print("-" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Data loading with num_workers=0 to avoid multiprocessing issues
    train_loaders, test_loader = load_mnist(num_clients=num_clients, alpha=alpha_dirichlet,
                                            batch_size=_tv_cfg['batch_size'],
                                            num_workers=_tv_cfg['num_workers'])

    if topology_type == 'mesh':
        from utils.topology import generate_mesh_topology
        neighbors = generate_mesh_topology(num_clients, degree=_tv_cfg['mesh_degree'])
    else:
        neighbors = generate_ring_topology(num_clients)

    # 节点划分
    num_byzantine = int(num_clients * byzantine_ratio)
    honest_nodes = list(range(num_clients - num_byzantine))
    byzantine_nodes = list(range(num_clients - num_byzantine, num_clients))

    print(f"Honest nodes: {len(honest_nodes)}")
    print(f"Byzantine nodes: {len(byzantine_nodes)}\n")

    # Initialize aggregators
    sama_agg = SAMAAggregator(
        alpha=_sama_cfg['alpha'],
        trust_layers=_sama_cfg.get('trust_layers', None)
    )
    balance_agg = BALANCEAggregator(alpha=_balance_cfg['alpha'],
                                    gamma=_balance_cfg['gamma'],
                                    kappa=_balance_cfg['kappa'])

    # Two independent model sets — each trained with its own aggregator
    models_sama = [SimpleCNN().to(device) for _ in range(num_clients)]
    models_balance = [SimpleCNN().to(device) for _ in range(num_clients)]

    # Sync initial weights so comparison is fair
    init_state = models_sama[0].state_dict()
    for i in range(num_clients):
        models_sama[i].load_state_dict(init_state)
        models_balance[i].load_state_dict(init_state)

    kappa_sama_history = []
    kappa_balance_history = []

    for t in range(num_rounds):
        # === SAMA track ===
        local_models_sama = []
        for i in range(num_clients):
            model = models_sama[i]
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
            elif attack_type == 'label_flipping':
                try:
                    data, target = next(iter(train_loaders[i]))
                    data, target = data.to(device), target.to(device)
                    target = (9 - target)
                    optimizer = torch.optim.SGD(model.parameters(), lr=_tv_cfg['lr'])
                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                except:
                    pass
            local_models_sama.append(model.state_dict())

        # === BALANCE track ===
        local_models_balance = []
        for i in range(num_clients):
            model = models_balance[i]
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
            elif attack_type == 'label_flipping':
                try:
                    data, target = next(iter(train_loaders[i]))
                    data, target = data.to(device), target.to(device)
                    target = (9 - target)
                    optimizer = torch.optim.SGD(model.parameters(), lr=_tv_cfg['lr'])
                    optimizer.zero_grad()
                    output = model(data)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
                except:
                    pass
            local_models_balance.append(model.state_dict())

        # Apply post-training attacks to both tracks
        if attack_type == 'gaussian':
            for byz_id in byzantine_nodes:
                malicious = OrderedDict()
                for key, param in local_models_sama[byz_id].items():
                    if isinstance(param, torch.Tensor):
                        malicious[key] = param + torch.randn_like(param) * _tv_cfg['gaussian_attack_std']
                    else:
                        malicious[key] = param
                local_models_sama[byz_id] = malicious

                malicious_b = OrderedDict()
                for key, param in local_models_balance[byz_id].items():
                    if isinstance(param, torch.Tensor):
                        malicious_b[key] = param + torch.randn_like(param) * _tv_cfg['gaussian_attack_std']
                    else:
                        malicious_b[key] = param
                local_models_balance[byz_id] = malicious_b
        elif attack_type == 'omniscient':
            omniscient = OmniscientAttack(amplification=_config['attack'].get('amplification', 2.0))
            honest_s = [local_models_sama[i] for i in honest_nodes]
            honest_b = [local_models_balance[i] for i in honest_nodes]
            for byz_id in byzantine_nodes:
                local_models_sama[byz_id] = omniscient.attack(honest_s)
                local_models_balance[byz_id] = omniscient.attack(honest_b)

        # Aggregate and estimate kappa for both tracks
        kappas_sama_round = []
        kappas_balance_round = []
        updated_sama = {}
        updated_balance = {}

        for i in honest_nodes:
            neighbor_ids = neighbors[i]
            honest_mask = [j in honest_nodes for j in neighbor_ids]

            # SAMA track
            own_s = local_models_sama[i]
            nbr_s = [local_models_sama[j] for j in neighbor_ids]
            agg_s = sama_agg.aggregate(own_s, nbr_s, t=t, T=num_rounds)
            final_s = sama_agg.final_update(own_s, agg_s)
            updated_sama[i] = final_s

            # BALANCE track
            own_b = local_models_balance[i]
            nbr_b = [local_models_balance[j] for j in neighbor_ids]
            agg_b, balance_stats = balance_agg.aggregate(own_b, nbr_b, t=t, T=num_rounds, return_stats=True)
            final_b = balance_agg.final_update(own_b, agg_b)
            updated_balance[i] = final_b

            # Estimate kappa (skip initial rounds)
            if t >= _tv_cfg['skip_initial_rounds']:
                # SAMA kappa
                own_vec_s = sama_agg.model_to_vector(own_s)
                nbr_vecs_s = [sama_agg.model_to_vector(m) for m in nbr_s]
                agg_vec_s = sama_agg.model_to_vector(agg_s)
                ks = KappaEstimator.estimate_kappa(own_vec_s, nbr_vecs_s, agg_vec_s, honest_mask, 'sama')
                if ks is not None and ks < _tv_cfg['kappa_filter_max']:
                    kappas_sama_round.append(ks)

                # BALANCE kappa
                if balance_stats.get('num_accepted', 0) > 0:
                    own_vec_b = balance_agg.model_to_vector(own_b)
                    nbr_vecs_b = [balance_agg.model_to_vector(m) for m in nbr_b]
                    agg_vec_b = balance_agg.model_to_vector(agg_b)
                    kb = KappaEstimator.estimate_kappa(own_vec_b, nbr_vecs_b, agg_vec_b, honest_mask, 'balance')
                    if kb is not None and kb < _tv_cfg['kappa_filter_max']:
                        kappas_balance_round.append(kb)

        # Update models with their own aggregation results
        for i in honest_nodes:
            if i in updated_sama:
                models_sama[i].load_state_dict(updated_sama[i])
            if i in updated_balance:
                models_balance[i].load_state_dict(updated_balance[i])

        if kappas_sama_round:
            kappa_sama_history.append(np.median(kappas_sama_round))
        if kappas_balance_round:
            kappa_balance_history.append(np.median(kappas_balance_round))
        else:
            # BALANCE may reject all neighbors when gamma is small;
            # in that case AGG=own_model, kappa is undefined.
            # Record 0.0 so the history stays aligned and the issue is visible.
            if t >= _tv_cfg['skip_initial_rounds']:
                kappa_balance_history.append(0.0)

        if (t + 1) % 50 == 0:
            recent_sama = np.mean(kappa_sama_history[-20:]) if kappa_sama_history else 0
            recent_balance = np.mean(kappa_balance_history[-20:]) if kappa_balance_history else 0
            balance_valid = sum(1 for k in kappa_balance_history[-20:] if k >= 0) if kappa_balance_history else 0
            print(f"Round {t+1}/{num_rounds}: kappa_SAMA={recent_sama:.4f}, "
                  f"kappa_BALANCE={recent_balance:.4f} ({balance_valid}/20 valid)")

    # 最终统计
    if not kappa_sama_history or not kappa_balance_history:
        print("ERROR: Insufficient kappa samples.")
        if not kappa_balance_history:
            print("  BALANCE produced 0 valid kappa estimates.")
            print(f"  gamma={_balance_cfg['gamma']} is likely too small — BALANCE rejects all neighbors.")
            print("  Try increasing gamma in configs/mnist.yaml (e.g., gamma: 3.0)")
        return None

    # κ=0.0 is valid (means perfect filtering), use all history directly
    kappa_sama_avg = np.mean(kappa_sama_history)
    kappa_balance_avg = np.mean(kappa_balance_history)
    kappa_sama_std = np.std(kappa_sama_history)
    kappa_balance_std = np.std(kappa_balance_history)

    print("=" * 80)
    print("Experiment Results")
    print("=" * 80)
    print(f"Hyperparameters: lr={_tv_cfg['lr']}, attack={attack_type}, "
          f"alpha_dirichlet={alpha_dirichlet}, topology={topology_type}")
    print(f"BALANCE: gamma={_balance_cfg['gamma']}, kappa_decay={_balance_cfg['kappa']}")
    print(f"SAMA: alpha={_sama_cfg['alpha']}")
    print("-" * 80)
    print(f"SAMA-DFL:  kappa = {kappa_sama_avg:.4f} +/- {kappa_sama_std:.4f}")
    print(f"BALANCE:   kappa = {kappa_balance_avg:.4f} +/- {kappa_balance_std:.4f}")
    if kappa_balance_avg > 1e-8:
        ratio = kappa_sama_avg / kappa_balance_avg
        improvement = (1 - ratio) * 100
        print(f"Ratio:     {ratio:.3f}")
        print(f"\nLemma 4.4 Verification: {'PASS - kappa_SAMA < kappa_BALANCE' if kappa_sama_avg < kappa_balance_avg else 'FAIL'}")
        print(f"Kappa reduction:   {improvement:.1f}%")
        print(f"Error term improvement: {(1 - ratio**2)*100:.1f}% (kappa^2 improvement)")
    else:
        improvement = float('nan')
        print("\nLemma 4.4 Verification: INCONCLUSIVE (BALANCE kappa ~ 0, likely gamma too small)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(kappa_sama_history, label='SAMA-DFL', linewidth=2, alpha=0.7)
    axes[0].plot(kappa_balance_history, label='BALANCE', linewidth=2, alpha=0.7)
    axes[0].axhline(kappa_sama_avg, color='C0', linestyle='--', alpha=0.5)
    axes[0].axhline(kappa_balance_avg, color='C1', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Training Round')
    axes[0].set_ylabel('Kappa Value')
    axes[0].set_title(f'Robustness Constant Kappa Over Time\n'
                      f'(n={num_clients}, f={int(byzantine_ratio*num_clients)}, '
                      f'attack={attack_type}, γ={_balance_cfg["gamma"]})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Right: bar chart comparison
    methods = ['SAMA-DFL', 'BALANCE']
    kappa_means = [kappa_sama_avg, kappa_balance_avg]
    kappa_stds = [kappa_sama_std, kappa_balance_std]

    x = np.arange(len(methods))
    axes[1].bar(x, kappa_means, yerr=kappa_stds, capsize=5, alpha=0.7, color=['C0', 'C1'])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(methods)
    axes[1].set_ylabel('Kappa Value')
    axes[1].set_title(f'Robustness Constant Kappa Comparison\n'
                      f'(α={_sama_cfg["alpha"]}, γ={_balance_cfg["gamma"]}, κ_decay={_balance_cfg["kappa"]})')
    axes[1].grid(True, alpha=0.3, axis='y')
    if not np.isnan(improvement):
        axes[1].text(0.5, max(kappa_means) * 0.8,
                    f'{improvement:.1f}% improvement',
                    ha='center', fontsize=12, color='green', fontweight='bold')
    else:
        axes[1].text(0.5, max(kappa_means) * 0.8,
                    'BALANCE kappa ~ 0\n(gamma too small)',
                    ha='center', fontsize=11, color='red', fontweight='bold')

    plt.tight_layout()

    # Save
    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f'kappa_measurement_{attack_type}.png'
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_dir / fname}")

    return {
        'kappa_sama': (kappa_sama_avg, kappa_sama_std),
        'kappa_balance': (kappa_balance_avg, kappa_balance_std),
        'improvement': improvement
    }


if __name__ == "__main__":
    results = run_kappa_measurement()
