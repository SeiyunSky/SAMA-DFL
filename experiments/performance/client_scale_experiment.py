"""
Client Scale Experiment
Sweep n ∈ {20, 30, 40}, 3 core methods (SAMA / BALANCE / SC-CLIP), MNIST.
Shows scalability w.r.t. number of clients.
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
import concurrent.futures
import threading

sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import (SAMAAggregator, BALANCEAggregator, SCCLIPAggregator,
                         FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator,
                         CoordMedianAggregator)
from models import SimpleCNN
from utils import load_mnist
from utils.topology import generate_mesh_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack, KrumAttack, TrimAttack


def _build_attack(attack_key, num_byzantine, config):
    atk = config['attack']
    if attack_key == 'none':
        from attacks import NoAttack
        return NoAttack()
    elif attack_key == 'gaussian':
        return GaussianAttack(std=atk['gaussian_std'])
    elif attack_key == 'label_flipping':
        return LabelFlippingAttack(num_classes=10)
    elif attack_key == 'omniscient':
        return OmniscientAttack(amplification=atk.get('amplification', 2.0))
    elif attack_key == 'krum_attack':
        return KrumAttack(num_byzantine=num_byzantine,
                          amplification=atk.get('amplification', 1.0))
    elif attack_key == 'trim_attack':
        return TrimAttack(num_byzantine=num_byzantine,
                          trim_ratio=atk.get('trim_ratio', 0.1))
    else:
        raise ValueError(f"Unknown attack: {attack_key}")


def _create_aggregator(method, config):
    if method == 'sama':
        return SAMAAggregator(
            alpha=config['sama']['alpha'],
            use_temperature=config['sama'].get('use_temperature', False),
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


def _train_one(config, method, device, neighbors=None, progress_queue=None, task_label=None):
    """单次训练，返回 final test accuracy (%)"""
    num_clients = config['federated']['num_clients']
    byz_ratio = config['federated']['byzantine_ratio']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['optimizer']['lr']
    mesh_degree = config['topology']['degree']

    train_loaders, test_loader = load_mnist(
        data_dir=config['data']['data_dir'],
        num_clients=num_clients,
        alpha=config['data']['non_iid_alpha'],
        batch_size=config['federated']['batch_size'],
        num_workers=config['federated'].get('num_workers', multiprocessing.cpu_count()),
    )

    if neighbors is None:
        # Mesh degree 不能超过 num_clients - 1
        degree = min(mesh_degree, num_clients - 1)
        neighbors = generate_mesh_topology(num_clients, degree=degree)

    num_byzantine = int(num_clients * byz_ratio)
    honest_nodes = list(range(num_clients - num_byzantine))
    byzantine_nodes = list(range(num_clients - num_byzantine, num_clients))

    attack_key = os.getenv('ATTACK_TYPE', config['attack']['type'])
    attack = _build_attack(attack_key, num_byzantine, config)
    aggregator = _create_aggregator(method, config)

    models = [SimpleCNN().to(device) for _ in range(num_clients)]

    for t in range(num_rounds):
        local_models = []
        for i in range(num_clients):
            model = models[i]
            if i in honest_nodes:
                model.train()
                for _ in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device), target.to(device)
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
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
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            local_models.append(model.state_dict())

        if not isinstance(attack, LabelFlippingAttack):
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

        models = [SimpleCNN().to(device) for _ in range(num_clients)]
        for i, state_dict in enumerate(updated_models):
            models[i].load_state_dict(state_dict)

        # 每轮结束向队列报告进度
        if progress_queue is not None and task_label is not None:
            progress_queue.put(f"[{task_label}] round {t+1}/{num_rounds}")

    honest_vecs = [aggregator.model_to_vector(updated_models[i]) for i in honest_nodes]
    honest_mean = torch.stack(honest_vecs).mean(dim=0)
    global_model = SimpleCNN().to(device)
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

    acc = 100.0 * correct / total

    # 完成时通知队列
    if progress_queue is not None and task_label is not None:
        progress_queue.put(f"[{task_label}] done accuracy={acc:.2f}%")

    return acc


def _progress_printer(queue, total_tasks, stop_event):
    """主进程监听线程：从队列读进度消息并打印到 stdout。"""
    done_count = 0
    while not stop_event.is_set() or not queue.empty():
        try:
            msg = queue.get(timeout=0.2)
        except Exception:
            continue
        print(msg, flush=True)
        if 'done' in msg:
            done_count += 1
            if done_count >= total_tasks:
                break


def run_client_scale_experiment(config_path=None):
    """
    Sweep n ∈ {20, 30, 40}, Byzantine ratio fixed at 20%.
    Methods: SAMA / BALANCE / SC-CLIP.
    """
    print("=" * 80)
    print("Client Scale Experiment: n ∈ {20, 30, 40}")
    print("=" * 80)

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    base_config['federated']['num_rounds'] = 150
    base_config['federated']['num_workers'] = multiprocessing.cpu_count()

    seed = base_config.get('experiment', {}).get('seed', 42)
    device = torch.device(base_config['experiment']['device'])
    methods = ['sama', 'balance', 'scclip', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'coord_median']
    method_labels = {
        'sama': 'SAMA', 'balance': 'BALANCE', 'scclip': 'SC-CLIP',
        'fedavg': 'FedAvg', 'krum': 'Krum', 'multi_krum': 'Multi-Krum',
        'trimmed_mean': 'Trim-Mean', 'coord_median': 'CoordMed',
    }
    n_values = [20, 30, 40]

    attack_key = os.getenv('ATTACK_TYPE', base_config['attack']['type'])
    byz_ratio = base_config['federated']['byzantine_ratio']
    noniid_alpha = base_config['data']['non_iid_alpha']

    results = {m: [] for m in methods}

    # Build all (n, method) tasks
    tasks = []
    task_neighbors = {}
    for n in n_values:
        torch.manual_seed(seed)
        np.random.seed(seed)
        degree = min(base_config['topology']['degree'], n - 1)
        shared_neighbors = generate_mesh_topology(n, degree=degree)
        task_neighbors[n] = shared_neighbors

        for method in methods:
            config = copy.deepcopy(base_config)
            config['federated']['num_clients'] = n
            label = f"n={n}/{method.upper()}"
            tasks.append((n, method, config, label))

    max_workers = min(len(tasks), 4)
    print(f"\n  Running {len(tasks)} tasks with {max_workers} workers in parallel...")

    mgr = multiprocessing.Manager()
    progress_queue = mgr.Queue()
    stop_event = threading.Event()

    printer = threading.Thread(
        target=_progress_printer,
        args=(progress_queue, len(tasks), stop_event),
        daemon=True,
    )
    printer.start()

    task_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(
                _train_one, config, method, device,
                task_neighbors[n], progress_queue, label
            ): (n, method)
            for n, method, config, label in tasks
        }
        for future in concurrent.futures.as_completed(future_to_key):
            n, method = future_to_key[future]
            acc = future.result()
            task_results[(n, method)] = acc

    stop_event.set()
    printer.join(timeout=2)

    for n in n_values:
        for method in methods:
            results[method].append(task_results[(n, method)])

    # ── 折线图 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(
        f"Scalability: Test Accuracy vs #Clients\n"
        f"MNIST | Attack={attack_key} | Byzantine={byz_ratio:.0%} | α={noniid_alpha}",
        fontsize=12, fontweight='bold'
    )

    colors = {
        'sama': '#1f77b4', 'balance': '#ff7f0e', 'scclip': '#2ca02c',
        'fedavg': '#d62728', 'krum': '#9467bd', 'multi_krum': '#8c564b',
        'trimmed_mean': '#e377c2', 'coord_median': '#7f7f7f',
    }
    markers = {
        'sama': 'o', 'balance': 's', 'scclip': '^',
        'fedavg': 'D', 'krum': 'v', 'multi_krum': '<',
        'trimmed_mean': 'P', 'coord_median': 'X',
    }

    for method in methods:
        ax.plot(n_values, results[method],
                label=method_labels[method],
                color=colors[method], marker=markers[method],
                linewidth=2, markersize=8)
        for x, y in zip(n_values, results[method]):
            ax.annotate(f'{y:.1f}%', (x, y), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=9)

    ax.set_xlabel('Number of Clients (n)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xticks(n_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"client_scale_{attack_key}_byz{int(byz_ratio*100)}_alpha{noniid_alpha}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_dir / fname}")

    # ── 控制台表格 ──────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'n':>6}", end="")
    for m in methods:
        print(f"  {method_labels[m]:>10}", end="")
    print()
    print("-" * 50)
    for i, n in enumerate(n_values):
        print(f"{n:>6}", end="")
        for m in methods:
            print(f"  {results[m][i]:>9.2f}%", end="")
        print()

    return results


if __name__ == "__main__":
    run_client_scale_experiment()
