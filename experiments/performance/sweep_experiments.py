"""
Parameter Sweep Experiments
C3: Byzantine ratio sweep
C4: Non-IID level sweep
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
from utils import load_mnist, generate_ring_topology
from utils.topology import generate_mesh_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack, KrumAttack, TrimAttack
from collections import OrderedDict


def create_aggregator(method, config):
    """Create aggregator by method name."""
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
            kappa=config['balance']['kappa']
        )
    elif method == 'scclip':
        return SCCLIPAggregator(
            alpha=config['scclip']['alpha'],
            clip_constant=config['scclip']['clip_constant']
        )
    elif method == 'fedavg':
        return FedAvgAggregator(alpha=config['fedavg']['alpha'])
    elif method == 'krum':
        return KrumAggregator(
            alpha=config['krum']['alpha'],
            byzantine_ratio=config['krum']['byzantine_ratio']
        )
    elif method == 'multi_krum':
        return KrumAggregator(
            alpha=config['multi_krum']['alpha'],
            multi_k=config['multi_krum']['multi_k'],
            byzantine_ratio=config['multi_krum']['byzantine_ratio']
        )
    elif method == 'trimmed_mean':
        return TrimmedMeanAggregator(
            alpha=config['trimmed_mean']['alpha'],
            trim_ratio=config['trimmed_mean']['trim_ratio']
        )
    elif method == 'coord_median':
        return CoordMedianAggregator(alpha=config['coord_median']['alpha'])
    else:
        raise ValueError(f"Unknown method: {method}")


def train_single_run(config, method, device, neighbors=None, progress_queue=None, task_label=None):
    """
    Run a single training experiment, return final accuracy.
    Simplified version of FederatedTrainer for sweep use.
    """
    # 支持字符串（spawn子进程跨进程传递），也支持 torch.device 对象
    if isinstance(device, str):
        device = torch.device(device)
    # spawn子进程里禁用DataLoader多进程，防止进程数爆炸
    num_workers = 0
    num_clients = config['federated']['num_clients']
    byz_ratio = config['federated']['byzantine_ratio']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['optimizer']['lr']

    # Data
    train_loaders, test_loader = load_mnist(
        data_dir=config['data']['data_dir'],
        num_clients=num_clients,
        alpha=config['data']['non_iid_alpha'],
        batch_size=config['federated']['batch_size'],
        num_workers=num_workers
    )

    # 使用外部传入的拓扑；否则生成
    if neighbors is None:
        topology_type = config['topology']['type']
        if topology_type == 'ring':
            neighbors = generate_ring_topology(num_clients)
        else:
            neighbors = generate_mesh_topology(num_clients, degree=config['topology']['degree'])

    # Node split
    num_byzantine = int(num_clients * byz_ratio)
    honest_nodes = set(range(num_clients - num_byzantine))
    byzantine_nodes = set(range(num_clients - num_byzantine, num_clients))

    # Attack
    attack_type = os.getenv('ATTACK_TYPE', config['attack']['type'])
    if attack_type == 'gaussian':
        attack = GaussianAttack(std=config['attack']['gaussian_std'])
    elif attack_type == 'label_flipping':
        attack = LabelFlippingAttack(num_classes=10)
    elif attack_type == 'omniscient':
        attack = OmniscientAttack(amplification=config['attack'].get('amplification', 2.0))
    elif attack_type == 'krum_attack':
        attack = KrumAttack(num_byzantine=num_byzantine,
                            amplification=config['attack'].get('amplification', 1.0))
    elif attack_type == 'trim_attack':
        attack = TrimAttack(num_byzantine=num_byzantine,
                            trim_ratio=config['attack'].get('trim_ratio', 0.1))
    else:
        attack = None

    # Models and aggregator
    models = [SimpleCNN().to(device) for _ in range(num_clients)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in models]
    aggregator = create_aggregator(method, config)

    # Training
    for t in range(num_rounds):
        local_vecs = [None] * num_clients
        for i in range(num_clients):
            model = models[i]
            optimizer = optimizers[i]

            if i in honest_nodes:
                model.train()
                for epoch in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            elif attack and isinstance(attack, LabelFlippingAttack):
                model.train()
                for epoch in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        target = attack.flip_labels(target)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()

            local_vecs[i] = aggregator.model_to_vector(models[i])

        # Post-training attacks
        if attack and not isinstance(attack, LabelFlippingAttack):
            honest_vecs = [local_vecs[i] for i in honest_nodes]
            if isinstance(attack, (OmniscientAttack, KrumAttack, TrimAttack)):
                for byz_id in byzantine_nodes:
                    local_vecs[byz_id] = attack.attack(honest_vecs)
            else:
                for byz_id in byzantine_nodes:
                    local_vecs[byz_id] = attack.attack(local_vecs[byz_id])

        # Aggregation
        all_vecs = torch.stack(local_vecs)
        updated_vecs = [None] * num_clients
        for i in range(num_clients):
            own_vec = local_vecs[i]
            neighbor_vecs = all_vecs[neighbors[i]]

            if i in honest_nodes:
                aggregated, agg_stats = aggregator.aggregate(
                    own_vec, neighbor_vecs, t=t, T=num_rounds, return_stats=True
                )
                avg_trust = agg_stats.get('avg_trust', None)
                final_vec = aggregator.final_update(own_vec, aggregated, avg_trust=avg_trust)
            else:
                final_vec = own_vec
            updated_vecs[i] = final_vec

        for i, vec in enumerate(updated_vecs):
            aggregator.load_from_vector(models[i], vec)

        # 每轮结束向队列报告进度
        if progress_queue is not None and task_label is not None:
            progress_queue.put(f"[{task_label}] round {t+1}/{num_rounds}")

    # Final evaluation on honest mean
    honest_vecs = [updated_vecs[i] for i in honest_nodes]
    honest_mean = torch.stack(honest_vecs).mean(dim=0)

    global_model = SimpleCNN().to(device)
    aggregator.load_from_vector(global_model, honest_mean)
    global_model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            pred = global_model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    acc = 100.0 * correct / total

    # 释放显存碎片
    torch.cuda.empty_cache()

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


def _run_parallel_with_progress(submit_fn, tasks, max_workers):
    """
    通用并行执行器，带进度队列。
    submit_fn(executor, task, queue) -> future
    tasks: list of任意结构，由 submit_fn 解包
    返回 {future: task} dict，调用方从 future.result() 取 acc
    """
    mgr = multiprocessing.Manager()
    progress_queue = mgr.Queue()
    stop_event = threading.Event()

    printer = threading.Thread(
        target=_progress_printer,
        args=(progress_queue, len(tasks), stop_event),
        daemon=True,
    )
    printer.start()

    future_to_task = {}
    mp_context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers,
                                                mp_context=mp_context) as executor:
        for task in tasks:
            future = submit_fn(executor, task, progress_queue)
            future_to_task[future] = task

        results = {}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            results[future] = future.result()

    stop_event.set()
    printer.join(timeout=2)
    return future_to_task, results


def run_byzantine_sweep(config_path=None):
    """
    C3: Byzantine ratio sweep
    Fixed: MNIST, alpha=0.1, Ring topology
    Sweep: byzantine_ratio in {0.1, 0.2, 0.3, 0.4}
    """
    print("=" * 80)
    print("Experiment C3: Byzantine Ratio Sweep")
    print("=" * 80)

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    device = torch.device(base_config['experiment']['device'])
    device_str = base_config['experiment']['device']
    methods = ['sama', 'balance', 'scclip', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'coord_median']
    byz_ratios = [0.1, 0.2, 0.3]
    results = {m: [] for m in methods}

    seed = base_config.get('experiment', {}).get('seed', 42)

    # Build all (byz_ratio, method) tasks
    tasks = []
    task_neighbors = {}
    for byz_ratio in byz_ratios:
        torch.manual_seed(seed)
        np.random.seed(seed)
        num_clients = base_config['federated']['num_clients']
        topology_type = base_config['topology']['type']
        if topology_type == 'ring':
            shared_neighbors = generate_ring_topology(num_clients)
        else:
            shared_neighbors = generate_mesh_topology(num_clients, degree=base_config['topology']['degree'])
        task_neighbors[byz_ratio] = shared_neighbors

        for method in methods:
            config = copy.deepcopy(base_config)
            config['federated']['byzantine_ratio'] = byz_ratio
            config['federated']['num_rounds'] = 150
            config['federated']['num_workers'] = 2
            label = f"byz{int(byz_ratio*100)}%/{method.upper()}"
            tasks.append((byz_ratio, method, config, label))

    max_workers = min(len(tasks), 4)

    def submit_fn(executor, task, queue):
        byz_ratio, method, config, label = task
        return executor.submit(
            train_single_run, config, method, device_str,
            task_neighbors[byz_ratio], queue, label
        )

    future_to_task, future_results = _run_parallel_with_progress(submit_fn, tasks, max_workers)

    task_results = {}
    for future, task in future_to_task.items():
        byz_ratio, method, config, label = task
        task_results[(byz_ratio, method)] = future_results[future]

    for byz_ratio in byz_ratios:
        for method in methods:
            results[method].append(task_results[(byz_ratio, method)])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    attack_type = base_config['attack']['type']
    noniid_alpha = base_config['data']['non_iid_alpha']
    fig.suptitle(f"Byzantine Robustness Curve | MNIST | Attack={attack_type} | α={noniid_alpha}",
                 fontsize=12, fontweight='bold')

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

    byz_pcts = [r * 100 for r in byz_ratios]
    for method in methods:
        label = method.upper().replace('_', '-')
        ax.plot(byz_pcts, results[method],
                label=label, color=colors[method],
                marker=markers[method], linewidth=2, markersize=8)

    ax.set_xlabel('Byzantine Ratio (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xticks(byz_pcts)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"byzantine_sweep_{attack_type}_alpha{noniid_alpha}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_dir / fname}")

    # Print table
    print("\n" + "=" * 60)
    print(f"{'Byz Ratio':>10}", end="")
    for m in methods:
        print(f"  {m.upper():>10}", end="")
    print()
    print("-" * 60)
    for i, ratio in enumerate(byz_ratios):
        print(f"{ratio:>10.0%}", end="")
        for m in methods:
            print(f"  {results[m][i]:>9.2f}%", end="")
        print()

    return results


def run_noniid_sweep(config_path=None):
    """
    C4: Non-IID level sweep
    Fixed: MNIST, byzantine_ratio=0.2, Ring topology
    Sweep: non_iid_alpha in {0.1, 0.3, 0.5, 1.0}
    """
    print("=" * 80)
    print("Experiment C4: Non-IID Level Sweep")
    print("=" * 80)

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    device = torch.device(base_config['experiment']['device'])
    device_str = base_config['experiment']['device']
    methods = ['sama', 'balance', 'scclip', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'coord_median']
    alpha_values = [0.1, 0.2, 0.3]
    results = {m: [] for m in methods}

    seed = base_config.get('experiment', {}).get('seed', 42)
    num_clients = base_config['federated']['num_clients']

    # Build all (alpha_val, method) tasks
    tasks = []
    task_neighbors = {}
    for alpha_val in alpha_values:
        torch.manual_seed(seed)
        np.random.seed(seed)
        topology_type = base_config['topology']['type']
        if topology_type == 'ring':
            shared_neighbors = generate_ring_topology(num_clients)
        else:
            shared_neighbors = generate_mesh_topology(num_clients, degree=base_config['topology']['degree'])
        task_neighbors[alpha_val] = shared_neighbors

        for method in methods:
            config = copy.deepcopy(base_config)
            config['data']['non_iid_alpha'] = alpha_val
            config['federated']['num_rounds'] = 150
            config['federated']['num_workers'] = 2
            label = f"α={alpha_val}/{method.upper()}"
            tasks.append((alpha_val, method, config, label))

    max_workers = min(len(tasks), 4)

    def submit_fn(executor, task, queue):
        alpha_val, method, config, label = task
        return executor.submit(
            train_single_run, config, method, device_str,
            task_neighbors[alpha_val], queue, label
        )

    future_to_task, future_results = _run_parallel_with_progress(submit_fn, tasks, max_workers)

    task_results = {}
    for future, task in future_to_task.items():
        alpha_val, method, config, label = task
        task_results[(alpha_val, method)] = future_results[future]

    for alpha_val in alpha_values:
        for method in methods:
            results[method].append(task_results[(alpha_val, method)])

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    attack_type = base_config['attack']['type']
    byz_ratio = base_config['federated']['byzantine_ratio']
    fig.suptitle(f"Non-IID Sensitivity | MNIST | Attack={attack_type} | Byzantine={byz_ratio:.0%}",
                 fontsize=12, fontweight='bold')

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
        label = method.upper().replace('_', '-')
        ax.plot(alpha_values, results[method],
                label=label, color=colors[method],
                marker=markers[method], linewidth=2, markersize=8)

    ax.set_xlabel('Dirichlet α (higher = more IID)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xticks(alpha_values)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"noniid_sweep_{attack_type}_byz{int(byz_ratio*100)}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_dir / fname}")

    # Print table
    print("\n" + "=" * 60)
    print(f"{'α':>10}", end="")
    for m in methods:
        print(f"  {m.upper():>10}", end="")
    print()
    print("-" * 60)
    for i, alpha_val in enumerate(alpha_values):
        print(f"{alpha_val:>10.1f}", end="")
        for m in methods:
            print(f"  {results[m][i]:>9.2f}%", end="")
        print()

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep', type=str, required=True,
                       choices=['byzantine', 'noniid'],
                       help='Sweep type: byzantine or noniid')
    args = parser.parse_args()

    if args.sweep == 'byzantine':
        run_byzantine_sweep()
    elif args.sweep == 'noniid':
        run_noniid_sweep()
