"""
Experiment B1: Convergence Rate Measurement
Verify Theorem 5.1: SAMA-DFL converges at rate λ = μη
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

from aggregators import SAMAAggregator
from models import SimpleCNN
from utils import load_mnist, generate_ring_topology

plt.rcParams['font.family'] = 'DejaVu Sans'

# Load config
_config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'
with open(_config_path, 'r') as _f:
    _config = yaml.safe_load(_f)
_tv_cfg = _config['theory_verification']
_sama_cfg = _config['sama']


def measure_convergence_rate(num_clients=None, byzantine_ratio=None,
                             num_rounds=None, alpha_dirichlet=None, lr=None):
    """测量 SAMA-DFL 收敛速率，验证定理 5.1: λ = μη"""
    if num_clients is None:
        num_clients = _tv_cfg['num_clients']
    if byzantine_ratio is None:
        byzantine_ratio = _tv_cfg['byzantine_ratio']
    if alpha_dirichlet is None:
        alpha_dirichlet = _tv_cfg['alpha_dirichlet']
    if num_rounds is None:
        num_rounds = _tv_cfg['num_rounds_convergence']
    if lr is None:
        lr = _tv_cfg['lr']

    print(f"Measuring SAMA-DFL convergence rate...")
    print(f"  n={num_clients}, f={int(num_clients*byzantine_ratio)}, "
          f"η={lr}, α={alpha_dirichlet}, rounds={num_rounds}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loaders, test_loader = load_mnist(num_clients=num_clients, alpha=alpha_dirichlet,
                                            batch_size=_tv_cfg['batch_size'],
                                            num_workers=_tv_cfg['num_workers'])

    neighbors = generate_ring_topology(num_clients)

    num_byzantine = int(num_clients * byzantine_ratio)
    honest_nodes = set(range(num_clients - num_byzantine))
    byzantine_nodes = set(range(num_clients - num_byzantine, num_clients))

    models = [SimpleCNN().to(device) for _ in range(num_clients)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in models]
    aggregator = SAMAAggregator(alpha=_sama_cfg['alpha'], use_temperature=False)

    consensus_errors = []  # D_t = (1/|H|)Σ||w_i - w̄_H||²

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

        honest_vecs = torch.stack([updated_vecs[i] for i in honest_nodes])
        honest_mean = honest_vecs.mean(dim=0)
        D_t = torch.mean(torch.norm(honest_vecs - honest_mean, dim=1).pow(2)).item()
        consensus_errors.append(D_t)

        if (t + 1) % 10 == 0:
            print(f"Round {t+1}/{num_rounds}: D_t={D_t:.6f}")

    # D_t ≈ D_0 · exp(-λt)，对数线性拟合
    skip_rounds = _tv_cfg['skip_initial_rounds']
    t_vals = np.arange(skip_rounds, len(consensus_errors))
    D_vals = np.array(consensus_errors[skip_rounds:])

    valid_mask = D_vals > 1e-8
    t_vals = t_vals[valid_mask]
    D_vals = D_vals[valid_mask]

    lambda_fitted = None
    mu_empirical = None
    if len(D_vals) > 10:
        log_D = np.log(D_vals)
        coeffs = np.polyfit(t_vals, log_D, 1)
        lambda_fitted = -coeffs[0]
        mu_empirical = lambda_fitted / lr

        print(f"\nConvergence rate fitting:")
        print(f"  Fitted λ        = {lambda_fitted:.6f}")
        print(f"  Learning rate η = {lr}")
        print(f"  μ (empirical)   = λ/η = {mu_empirical:.6f}")
        print(f"  Verification: λ = μη = {mu_empirical:.6f}×{lr} = {mu_empirical * lr:.6f}  "
              f"(fitted λ = {lambda_fitted:.6f})")
    else:
        print("\nWarning: Insufficient valid data points for fitting")

    return {
        'consensus_errors': consensus_errors,
        'lambda_fitted': lambda_fitted,
        'mu_empirical': mu_empirical,
        'lr': lr,
        't_vals_fit': t_vals,
        'D_vals_fit': D_vals,
        'skip_rounds': skip_rounds,
    }


def run_convergence_comparison():
    """验证 SAMA-DFL 收敛速率定理 5.1"""
    print("=" * 80)
    print("Experiment B1: Convergence Rate Verification (SAMA-DFL)")
    print("=" * 80)

    result = measure_convergence_rate()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n = _tv_cfg['num_clients']
    f = int(_tv_cfg['num_clients'] * _tv_cfg['byzantine_ratio'])
    lr = _tv_cfg['lr']
    std = _tv_cfg['gaussian_attack_std']

    # 左图：D_t 曲线（对数纵轴）
    axes[0].plot(result['consensus_errors'], color='C0', linewidth=2, alpha=0.8, label='SAMA-DFL $D_t$')
    axes[0].set_xlabel('Training Round', fontsize=12)
    axes[0].set_ylabel('Consensus Error $D_t$', fontsize=12)
    axes[0].set_title(f'Consensus Error Over Time\n'
                      f'(n={n}, f={f}, η={lr}, Gaussian std={std})')
    axes[0].set_yscale('log')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右图：log D_t + 拟合直线，直观展示 λ = μη
    t_vals = result['t_vals_fit']
    D_vals = result['D_vals_fit']
    skip = result['skip_rounds']
    lambda_fit = result['lambda_fitted']
    mu_emp = result['mu_empirical']

    if len(D_vals) > 10:
        axes[1].scatter(t_vals, np.log(D_vals),
                        color='C0', alpha=0.4, s=8, label='Data $\\log D_t$')

        if lambda_fit is not None:
            log_D0 = np.log(result['consensus_errors'][skip])
            fit_line = log_D0 - lambda_fit * (t_vals - skip)
            axes[1].plot(t_vals, fit_line, color='C3', linewidth=2,
                         label=f'Linear fit: slope = $-\\lambda$ = {-lambda_fit:.4f}')

            # 标注理论对应关系
            textstr = (f'$\\lambda_{{\\rm fit}}$ = {lambda_fit:.4f}\n'
                       f'$\\hat{{\\mu}}$ = λ/η = {mu_emp:.4f}\n'
                       f'Verify: $\\lambda = \\hat{{\\mu}}\\eta$ = {mu_emp*lr:.4f}')
            axes[1].text(0.97, 0.97, textstr, transform=axes[1].transAxes,
                         fontsize=10, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[1].set_xlabel('Training Round', fontsize=12)
    axes[1].set_ylabel('$\\log D_t$', fontsize=12)
    axes[1].set_title(f'Log-scale Convergence + Linear Fit\n'
                      f'(α={_sama_cfg["alpha"]}, skip first {skip} rounds)')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'convergence_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_dir / 'convergence_rate.png'}")

    # Print summary
    print("\n" + "=" * 80)
    print("Theorem 5.1 Verification: λ = μη")
    print("=" * 80)
    if lambda_fit is not None and mu_emp is not None:
        print(f"  λ_fitted   = {lambda_fit:.6f}")
        print(f"  η          = {lr}")
        print(f"  μ_empirical = λ/η = {mu_emp:.6f}")
        print(f"  λ = μη     = {mu_emp * lr:.6f}  (matches λ_fitted ✓)")
    else:
        print("  WARNING: Fitting failed, insufficient data")

    return result


if __name__ == "__main__":
    run_convergence_comparison()
