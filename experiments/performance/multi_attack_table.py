"""
Multi-Attack Comparison Table
8 methods × 6 attack types on MNIST / CIFAR-10, outputs:
  - Console table (plain + LaTeX)
  - Heatmap PNG
  - Per-attack convergence curves (6 PNGs, 8 method lines each)
  - Per-method convergence curves (8 PNGs, 6 attack lines each)
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import yaml
import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import (SAMAAggregator, BALANCEAggregator, SCCLIPAggregator,
                         FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator,
                         CoordMedianAggregator)
from models import SimpleCNN
from utils import load_mnist, load_cifar10
from utils.topology import generate_mesh_topology
from attacks import (NoAttack, GaussianAttack, LabelFlippingAttack,
                     OmniscientAttack, KrumAttack, TrimAttack)


def _build_attack(key, num_byzantine, config):
    atk = config['attack']
    if key == 'none':
        return NoAttack()
    elif key == 'gaussian':
        return GaussianAttack(std=atk['gaussian_std'])
    elif key == 'label_flipping':
        return LabelFlippingAttack(num_classes=10)
    elif key == 'omniscient':
        return OmniscientAttack(amplification=atk.get('amplification', 2.0))
    elif key == 'krum_attack':
        return KrumAttack(num_byzantine=num_byzantine,
                          amplification=atk.get('amplification', 1.0))
    elif key == 'trim_attack':
        return TrimAttack(num_byzantine=num_byzantine,
                          trim_ratio=atk.get('trim_ratio', 0.1))
    else:
        raise ValueError(f"Unknown attack: {key}")


ATTACK_KEYS = ['none', 'gaussian', 'label_flipping', 'omniscient', 'krum_attack', 'trim_attack']
ATTACK_LABELS = ['No Attack', 'Gaussian', 'Label Flip', 'Omniscient', 'Krum Atk', 'Trim Atk']

METHOD_KEYS = ['sama', 'balance', 'scclip', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'coord_median']
METHOD_LABELS = ['SAMA', 'BALANCE', 'SC-CLIP', 'FedAvg', 'Krum', 'Multi-Krum', 'Trim-Mean', 'CoordMed']

METHOD_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]
METHOD_LINESTYLES = ['-', '-', '-', '--', '--', '--', '-.', '-.']

ATTACK_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
]
ATTACK_LINESTYLES = ['-', '--', '-.', ':', '-', '--']


def _create_aggregator(method, config):
    if method == 'sama':
        return SAMAAggregator(
            alpha=config['sama']['alpha'],
            use_temperature=config['sama'].get('use_temperature', False),
            tau_max=config['sama'].get('tau_max', 1.0),
            tau_min=config['sama'].get('tau_min', 0.01),
            trust_layers=config['sama'].get('trust_layers', None),
        )
    elif method == 'balance':
        return BALANCEAggregator(
            alpha=config['balance']['alpha'],
            gamma=config['balance']['gamma'],
            kappa=config['balance']['kappa'],
        )
    elif method == 'scclip':
        return SCCLIPAggregator(
            alpha=config['scclip']['alpha'],
            clip_constant=config['scclip']['clip_constant'],
        )
    elif method == 'fedavg':
        return FedAvgAggregator(alpha=config['fedavg']['alpha'])
    elif method == 'krum':
        return KrumAggregator(
            alpha=config['krum']['alpha'],
            byzantine_ratio=config['krum']['byzantine_ratio'],
        )
    elif method == 'multi_krum':
        return KrumAggregator(
            alpha=config['multi_krum']['alpha'],
            multi_k=config['multi_krum']['multi_k'],
            byzantine_ratio=config['multi_krum']['byzantine_ratio'],
        )
    elif method == 'trimmed_mean':
        return TrimmedMeanAggregator(
            alpha=config['trimmed_mean']['alpha'],
            trim_ratio=config['trimmed_mean']['trim_ratio'],
        )
    elif method == 'coord_median':
        return CoordMedianAggregator(alpha=config['coord_median']['alpha'])
    else:
        raise ValueError(f"Unknown method: {method}")


def _train_one(config, method, attack_key, device, dataset='mnist', neighbors=None):
    """单次训练，返回 (最终accuracy, accuracy历史列表)"""
    num_clients = config['federated']['num_clients']
    byz_ratio = config['federated']['byzantine_ratio']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['optimizer']['lr']
    momentum = config['optimizer'].get('momentum', 0.0)
    weight_decay = config['optimizer'].get('weight_decay', 0.0)
    log_interval = config['logging']['log_interval']

    num_workers = config['federated'].get('num_workers', multiprocessing.cpu_count())

    if dataset == 'cifar10':
        train_loaders, test_loader = load_cifar10(
            data_dir=config['data']['data_dir'],
            num_clients=num_clients,
            alpha=config['data']['non_iid_alpha'],
            batch_size=config['federated']['batch_size'],
            num_workers=num_workers,
        )
        model_kwargs = {'num_classes': 10, 'in_channels': 3}
    else:
        train_loaders, test_loader = load_mnist(
            data_dir=config['data']['data_dir'],
            num_clients=num_clients,
            alpha=config['data']['non_iid_alpha'],
            batch_size=config['federated']['batch_size'],
            num_workers=num_workers,
        )
        model_kwargs = {}

    # 使用外部传入的拓扑（保证所有run共享同一拓扑），否则临时生成
    if neighbors is None:
        neighbors = generate_mesh_topology(num_clients, degree=config['topology']['degree'])

    num_byzantine = int(num_clients * byz_ratio)
    honest_nodes = list(range(num_clients - num_byzantine))
    byzantine_nodes = list(range(num_clients - num_byzantine, num_clients))

    attack = _build_attack(attack_key, num_byzantine, config)
    aggregator = _create_aggregator(method, config)

    models = [SimpleCNN(**model_kwargs).to(device) for _ in range(num_clients)]
    acc_history = []

    for t in range(num_rounds):
        local_models = []
        for i in range(num_clients):
            model = models[i]
            if i in honest_nodes:
                model.train()
                for _ in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device), target.to(device)
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                                    momentum=momentum,
                                                    weight_decay=weight_decay)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            elif isinstance(attack, LabelFlippingAttack):
                model.train()
                for _ in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device), target.to(device)
                        target = attack.flip_labels(target)
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                                    momentum=momentum,
                                                    weight_decay=weight_decay)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            local_models.append(model.state_dict())

        if not isinstance(attack, (NoAttack, LabelFlippingAttack)):
            honest_models = [local_models[i] for i in honest_nodes]
            if isinstance(attack, (OmniscientAttack, KrumAttack, TrimAttack)):
                for byz_id in byzantine_nodes:
                    local_models[byz_id] = attack.attack(honest_models)
            else:
                for byz_id in byzantine_nodes:
                    local_models[byz_id] = attack.attack(local_models[byz_id])

        updated_models = []
        for i in range(num_clients):
            own_model = local_models[i]
            neighbor_models = [local_models[j] for j in neighbors[i]]
            if i in honest_nodes:
                aggregated, agg_stats = aggregator.aggregate(
                    own_model, neighbor_models, t=t, T=num_rounds, return_stats=True
                )
                avg_trust = agg_stats.get('avg_trust', None)
                final = aggregator.final_update(own_model, aggregated, avg_trust=avg_trust)
            else:
                final = own_model
            updated_models.append(final)

        models = [SimpleCNN(**model_kwargs).to(device) for _ in range(num_clients)]
        for i, state_dict in enumerate(updated_models):
            models[i].load_state_dict(state_dict)

        # 周期评估
        if (t + 1) % log_interval == 0:
            honest_vecs = [aggregator.model_to_vector(updated_models[i]) for i in honest_nodes]
            honest_mean = torch.stack(honest_vecs).mean(dim=0)
            global_model = SimpleCNN(**model_kwargs).to(device)
            global_model.load_state_dict(
                aggregator.vector_to_model(honest_mean, global_model.state_dict())
            )
            global_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    pred = global_model(data).argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            acc_history.append(100.0 * correct / total)

    final_acc = acc_history[-1] if acc_history else 0.0
    return final_acc, acc_history


def _worker(args):
    """顶层 worker，可被 ProcessPoolExecutor pickle。"""
    m_idx, a_idx, config, method, attack_key, device_str, dataset, neighbors = args
    # 子进程里固定 per-task seed，保证每个 (method, attack) 可复现
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed + m_idx * 100 + a_idx)
    np.random.seed(seed + m_idx * 100 + a_idx)
    # 子进程里 DataLoader 不开额外 workers，防止进程数爆炸
    config['federated']['num_workers'] = 0
    device = torch.device(device_str)
    acc, history = _train_one(config, method, attack_key, device,
                              dataset=dataset, neighbors=neighbors)
    return m_idx, a_idx, acc, history


def _run_table(base_config, dataset_name, save_dir):
    """
    通用运行逻辑：8方法 × 6攻击，并行输出全部图表。
    dataset_name: 'mnist' | 'cifar10'，用于文件名和标题。
    """
    # 固定随机种子，保证可复现
    seed = base_config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device_str = base_config['experiment']['device']
    byz_ratio = base_config['federated']['byzantine_ratio']
    noniid_alpha = base_config['data']['non_iid_alpha']
    log_interval = base_config['logging']['log_interval']
    num_rounds = base_config['federated']['num_rounds']
    num_clients = base_config['federated']['num_clients']

    # 预生成一次拓扑，所有 (method, attack) 共享
    shared_neighbors = generate_mesh_topology(num_clients, degree=base_config['topology']['degree'])

    results = np.zeros((len(METHOD_KEYS), len(ATTACK_KEYS)))
    histories = [[None] * len(ATTACK_KEYS) for _ in range(len(METHOD_KEYS))]

    # 并发数：每个 CUDA context 约 400MB，11GB 显存安全上限约 8 个
    # 可通过环境变量 TABLE_WORKERS 覆盖（如 TABLE_WORKERS=4）
    max_workers = int(os.getenv('TABLE_WORKERS', min(multiprocessing.cpu_count(), 8)))

    tasks = []
    for a_idx, attack_key in enumerate(ATTACK_KEYS):
        for m_idx, method in enumerate(METHOD_KEYS):
            config = copy.deepcopy(base_config)
            tasks.append((m_idx, a_idx, config, method, attack_key,
                          device_str, dataset_name, shared_neighbors))

    total = len(tasks)
    pbar = tqdm(total=total, desc=f"Running {dataset_name.upper()} table")

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            m_idx, a_idx, acc, history = future.result()
            results[m_idx, a_idx] = acc
            histories[m_idx][a_idx] = history
            method = METHOD_KEYS[m_idx]
            attack = ATTACK_KEYS[a_idx]
            pbar.set_postfix({'method': method, 'attack': attack, 'acc': f'{acc:.1f}%'})
            pbar.update(1)

    pbar.close()

    tag = f"{dataset_name}_byz{int(byz_ratio*100)}_alpha{noniid_alpha}"
    ds_upper = dataset_name.upper()

    # ── 热力图 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(results, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)')
    ax.set_xticks(range(len(ATTACK_KEYS)))
    ax.set_xticklabels(ATTACK_LABELS, fontsize=11)
    ax.set_yticks(range(len(METHOD_KEYS)))
    ax.set_yticklabels(METHOD_LABELS, fontsize=11)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Aggregation Method', fontsize=12)
    ax.set_title(
        f'{ds_upper}: Accuracy (%) by Method × Attack\n'
        f'Byzantine={byz_ratio:.0%}, Dirichlet α={noniid_alpha}, Mesh degree=6',
        fontsize=12, fontweight='bold'
    )
    for m_idx in range(len(METHOD_KEYS)):
        for a_idx in range(len(ATTACK_KEYS)):
            val = results[m_idx, a_idx]
            color = 'black' if 30 < val < 80 else 'white'
            ax.text(a_idx, m_idx, f'{val:.1f}', ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')
    plt.tight_layout()
    fig_path = save_dir / f'heatmap_{tag}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {fig_path.name}")

    # ── per-attack 折线图（6张，每张8条方法曲线）──────────────
    round_ticks = list(range(log_interval, num_rounds + 1, log_interval))
    for a_idx, (attack_key, attack_label) in enumerate(zip(ATTACK_KEYS, ATTACK_LABELS)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, (method_label, color, ls) in enumerate(
                zip(METHOD_LABELS, METHOD_COLORS, METHOD_LINESTYLES)):
            hist = histories[m_idx][a_idx]
            if hist:
                ax.plot(round_ticks[:len(hist)], hist,
                        label=method_label, color=color, linestyle=ls,
                        linewidth=1.8, marker='o', markersize=3)
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(
            f'{ds_upper} — {attack_label}\n'
            f'Byzantine={byz_ratio:.0%}, Dirichlet α={noniid_alpha}, Mesh d=6',
            fontsize=12, fontweight='bold'
        )
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9, ncol=2, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        curve_path = save_dir / f'acc_per_attack_{attack_key}_{tag}.png'
        plt.savefig(curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-attack curve saved: {curve_path.name}")

    # ── per-method 折线图（8张，每张6条攻击曲线）──────────────
    for m_idx, (method_key, method_label) in enumerate(zip(METHOD_KEYS, METHOD_LABELS)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for a_idx, (attack_label, color, ls) in enumerate(
                zip(ATTACK_LABELS, ATTACK_COLORS, ATTACK_LINESTYLES)):
            hist = histories[m_idx][a_idx]
            if hist:
                ax.plot(round_ticks[:len(hist)], hist,
                        label=attack_label, color=color, linestyle=ls,
                        linewidth=1.8, marker='o', markersize=3)
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(
            f'{ds_upper} — {method_label}\n'
            f'Byzantine={byz_ratio:.0%}, Dirichlet α={noniid_alpha}, Mesh d=6',
            fontsize=12, fontweight='bold'
        )
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9, ncol=2, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        method_path = save_dir / f'acc_per_method_{method_key}_{tag}.png'
        plt.savefig(method_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Per-method curve saved: {method_path.name}")

    # ── 控制台表格 ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Method':>12}", end="")
    for label in ATTACK_LABELS:
        print(f"  {label:>10}", end="")
    print()
    print("-" * 80)
    for m_idx, label in enumerate(METHOD_LABELS):
        print(f"{label:>12}", end="")
        for a_idx in range(len(ATTACK_KEYS)):
            print(f"  {results[m_idx, a_idx]:>9.2f}%", end="")
        print()

    # ── LaTeX 表格 ──────────────────────────────────────────
    print(f"\n% LaTeX table ({ds_upper}):")
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{{ds_upper} Test Accuracy (\\%) under 6 Attack Types "
          f"(Byzantine={byz_ratio:.0%}, $\\alpha={noniid_alpha}$, Mesh $d=6$)}}")
    print(f"\\label{{tab:multi-attack-{dataset_name}}}")
    cols = "l" + "c" * len(ATTACK_KEYS)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    print("Method & " + " & ".join(ATTACK_LABELS) + " \\\\")
    print("\\midrule")
    best_per_col = results.argmax(axis=0)
    for m_idx, label in enumerate(METHOD_LABELS):
        row_parts = []
        for a_idx in range(len(ATTACK_KEYS)):
            val_str = f"{results[m_idx, a_idx]:.1f}"
            if best_per_col[a_idx] == m_idx:
                row_parts.append(f"\\textbf{{{val_str}}}")
            else:
                row_parts.append(val_str)
        print(f"{label} & " + " & ".join(row_parts) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    return results


def run_multi_attack_table(config_path=None):
    """Run 8-method × 6-attack full comparison table on MNIST."""
    print("=" * 80)
    print("Multi-Attack Comparison Table (8 methods × 6 attacks, MNIST)")
    print("=" * 80)

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    base_config['federated']['num_rounds'] = 150
    base_config['federated']['num_workers'] = multiprocessing.cpu_count()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)

    return _run_table(base_config, 'mnist', save_dir)


def run_cifar10_attack_table(config_path=None):
    """Run 8-method × 6-attack full comparison table on CIFAR-10."""
    print("=" * 80)
    print("Multi-Attack Comparison Table (8 methods × 6 attacks, CIFAR-10)")
    print("=" * 80)

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'cifar10.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    base_config['federated']['num_rounds'] = 200
    base_config['federated']['num_workers'] = multiprocessing.cpu_count()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)

    return _run_table(base_config, 'cifar10', save_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10'])
    args = parser.parse_args()
    if args.dataset == 'cifar10':
        run_cifar10_attack_table()
    else:
        run_multi_attack_table()
