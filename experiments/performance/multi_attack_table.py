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
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm as _tqdm_base
import sys as _sys
def tqdm(*a, **kw):
    kw.setdefault('file', _sys.stdout)
    kw.setdefault('ascii', True)
    kw.setdefault('mininterval', 2.0)
    kw.setdefault('miniters', 1)
    return _tqdm_base(*a, **kw)
from collections import OrderedDict

sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import (SAMAAggregator, BALANCEAggregator, SCCLIPAggregator,
                         FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator,
                         CoordMedianAggregator)
from models import SimpleCNN
from utils import load_mnist, load_cifar10, load_mnist_gpu, load_cifar10_gpu
from utils.topology import generate_mesh_topology
from attacks import (NoAttack, GaussianAttack, LabelFlippingAttack,
                     OmniscientAttack, KrumAttack, TrimAttack, BrokenNodeAttack)


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
    elif key == 'broken_node':
        return BrokenNodeAttack()
    else:
        raise ValueError(f"Unknown attack: {key}")


ATTACK_KEYS = ['none', 'gaussian', 'label_flipping', 'omniscient', 'krum_attack', 'trim_attack', 'broken_node']
ATTACK_LABELS = ['No Attack', 'Gaussian', 'Label Flip', 'Omniscient', 'Krum Atk', 'Trim Atk', 'Broken Node']

METHOD_KEYS = ['sama', 'balance', 'scclip', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'coord_median']
METHOD_LABELS = ['SAMA', 'BALANCE', 'SC-CLIP', 'FedAvg', 'Krum', 'Multi-Krum', 'Trim-Mean', 'CoordMed']

METHOD_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
]
METHOD_LINESTYLES = ['-', '-', '-', '--', '--', '--', '-.', '-.']

ATTACK_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf',
]
ATTACK_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.']


def _create_aggregator(method, config, model_template=None):
    if method == 'sama':
        return SAMAAggregator(
            alpha=config['sama']['alpha'],
            use_temperature=config['sama'].get('use_temperature', False),
            tau_max=config['sama'].get('tau_max', 1.0),
            tau_min=config['sama'].get('tau_min', 0.01),
            trust_layers=config['sama'].get('trust_layers', None),
            model_template=model_template,
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


def _train_one(config, method, attack_key, device, dataset='mnist', neighbors=None,
               progress_queue=None, task_label=None, log_file=None):
    """单次训练，返回 (最终accuracy, accuracy历史列表)"""
    num_clients = config['federated']['num_clients']
    byz_ratio = config['federated']['byzantine_ratio']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['optimizer']['lr']
    momentum = config['optimizer'].get('momentum', 0.0)
    weight_decay = config['optimizer'].get('weight_decay', 0.0)
    log_interval = config['logging']['log_interval']

    num_workers = config['federated'].get('num_workers', 2)
    use_gpu_loader = bool(int(os.getenv('GPU_LOADER', '1')))  # 默认开

    if dataset == 'cifar10':
        if use_gpu_loader:
            train_loaders, test_loader = load_cifar10_gpu(
                data_dir=config['data']['data_dir'],
                num_clients=num_clients,
                alpha=config['data']['non_iid_alpha'],
                batch_size=config['federated']['batch_size'],
                device=str(device),
            )
        else:
            train_loaders, test_loader = load_cifar10(
                data_dir=config['data']['data_dir'],
                num_clients=num_clients,
                alpha=config['data']['non_iid_alpha'],
                batch_size=config['federated']['batch_size'],
                num_workers=num_workers,
            )
        model_kwargs = {'num_classes': 10, 'in_channels': 3}
    else:
        if use_gpu_loader:
            train_loaders, test_loader = load_mnist_gpu(
                data_dir=config['data']['data_dir'],
                num_clients=num_clients,
                alpha=config['data']['non_iid_alpha'],
                batch_size=config['federated']['batch_size'],
                device=str(device),
            )
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
    honest_nodes = set(range(num_clients - num_byzantine))
    byzantine_nodes = set(range(num_clients - num_byzantine, num_clients))

    models = [SimpleCNN(**model_kwargs).to(device) for _ in range(num_clients)]
    eval_model = SimpleCNN(**model_kwargs).to(device)
    attack = _build_attack(attack_key, num_byzantine, config)
    aggregator = _create_aggregator(method, config, model_template=models[0])
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum,
                                  weight_decay=weight_decay) for m in models]
    acc_history = []

    for t in range(num_rounds):
        local_vecs = [None] * num_clients
        for i in range(num_clients):
            model = models[i]
            optimizer = optimizers[i]
            # attack=NoAttack 时，所有节点都视为诚实节点正常训练（byzantine_ratio 失效）
            is_training_node = (i in honest_nodes) or isinstance(attack, NoAttack)
            if is_training_node:
                model.train()
                for _ in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            elif isinstance(attack, LabelFlippingAttack):
                model.train()
                for _ in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        target = attack.flip_labels(target)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            local_vecs[i] = aggregator.model_to_vector(models[i])

        if not isinstance(attack, (NoAttack, LabelFlippingAttack, BrokenNodeAttack)):
            honest_vecs = [local_vecs[i] for i in honest_nodes]
            if isinstance(attack, (OmniscientAttack, KrumAttack, TrimAttack)):
                for byz_id in byzantine_nodes:
                    local_vecs[byz_id] = attack.attack(honest_vecs)
            else:
                for byz_id in byzantine_nodes:
                    local_vecs[byz_id] = attack.attack(local_vecs[byz_id])

        all_vecs = torch.stack(local_vecs)
        updated_vecs = [None] * num_clients
        for i in range(num_clients):
            own_vec = local_vecs[i]
            neighbor_vecs = all_vecs[neighbors[i]]
            # attack=NoAttack 时所有节点都参与聚合
            participates = (i in honest_nodes) or isinstance(attack, NoAttack)
            if participates:
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

        if progress_queue is not None and task_label is not None:
            progress_queue.put(f"[{task_label}] round {t+1}/{num_rounds}")
        if (t + 1) % log_interval == 0:
            # attack=NoAttack 时 eval 用全节点平均；否则只用 honest 节点平均
            if isinstance(attack, NoAttack):
                eval_indices = list(range(num_clients))
            else:
                eval_indices = list(honest_nodes)
            eval_vecs = [updated_vecs[i] for i in eval_indices]
            eval_mean = torch.stack(eval_vecs).mean(dim=0)
            aggregator.load_from_vector(eval_model, eval_mean)
            eval_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                    pred = eval_model(data).argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
            acc_history.append(100.0 * correct / total)
            if log_file is not None:
                with open(log_file, 'a') as f:
                    f.write(f"round={t+1}/{num_rounds} acc={acc_history[-1]:.2f}%\n")

    final_acc = acc_history[-1] if acc_history else 0.0

    # 显式释放 GPU 显存：dataloader（含预加载的 tensor）、模型、优化器
    del train_loaders, test_loader
    del models, eval_model, optimizers, aggregator
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return final_acc, acc_history


def _worker(args):
    """顶层 worker，可被 ProcessPoolExecutor pickle。"""
    m_idx, a_idx, config, method, attack_key, device_str, dataset, neighbors, progress_queue = args
    # 子进程里固定 per-task seed，保证每个 (method, attack) 可复现
    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed + m_idx * 100 + a_idx)
    np.random.seed(seed + m_idx * 100 + a_idx)
    # 子进程里 DataLoader 不开额外 workers，防止进程数爆炸
    config['federated']['num_workers'] = 0
    device = torch.device(device_str)
    task_label = f"{method.upper()}/{attack_key}"

    log_dir = Path(__file__).parent.parent.parent / 'results' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{dataset}_{method}_{attack_key}.log"

    acc, history = _train_one(config, method, attack_key, device,
                              dataset=dataset, neighbors=neighbors,
                              progress_queue=progress_queue, task_label=task_label,
                              log_file=log_file)
    # 子进程退出前再清一遍，确保 spawn 进程释放干净
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if progress_queue is not None:
        progress_queue.put(f"[{task_label}] done accuracy={acc:.2f}%")
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

    # 允许通过环境变量 TABLE_ATTACKS 限定要跑的攻击子集（逗号分隔）
    attack_filter = os.getenv('TABLE_ATTACKS', '').strip()
    if attack_filter:
        wanted = [k.strip() for k in attack_filter.split(',') if k.strip()]
        attack_indices = [ATTACK_KEYS.index(k) for k in wanted if k in ATTACK_KEYS]
        active_attack_keys = [ATTACK_KEYS[i] for i in attack_indices]
        active_attack_labels = [ATTACK_LABELS[i] for i in attack_indices]
    else:
        attack_indices = list(range(len(ATTACK_KEYS)))
        active_attack_keys = ATTACK_KEYS
        active_attack_labels = ATTACK_LABELS

    # 允许通过环境变量 TABLE_METHODS 限定要跑的算法子集（逗号分隔）
    method_filter = os.getenv('TABLE_METHODS', '').strip()
    if method_filter:
        wanted_m = [k.strip() for k in method_filter.split(',') if k.strip()]
        method_indices = [METHOD_KEYS.index(k) for k in wanted_m if k in METHOD_KEYS]
        active_method_keys = [METHOD_KEYS[i] for i in method_indices]
        active_method_labels = [METHOD_LABELS[i] for i in method_indices]
        active_method_colors = [METHOD_COLORS[i] for i in method_indices]
        active_method_linestyles = [METHOD_LINESTYLES[i] for i in method_indices]
    else:
        method_indices = list(range(len(METHOD_KEYS)))
        active_method_keys = METHOD_KEYS
        active_method_labels = METHOD_LABELS
        active_method_colors = METHOD_COLORS
        active_method_linestyles = METHOD_LINESTYLES

    results = np.zeros((len(active_method_keys), len(active_attack_keys)))
    histories = [[None] * len(active_attack_keys) for _ in range(len(active_method_keys))]

    max_workers = int(os.getenv('TABLE_WORKERS', base_config.get('experiment', {}).get('parallel_workers', 4)))

    mgr = multiprocessing.Manager()
    progress_queue = mgr.Queue()

    tasks = []
    for a_idx, attack_key in enumerate(active_attack_keys):
        for m_idx, method in enumerate(active_method_keys):
            config = copy.deepcopy(base_config)
            tasks.append((m_idx, a_idx, config, method, attack_key,
                          device_str, dataset_name, shared_neighbors, progress_queue))

    total = len(tasks)
    pbar = tqdm(total=total, desc=f"Running {dataset_name.upper()} table")

    # 监听线程：把孙进程的进度消息打到 stdout
    stop_event = threading.Event()
    def _printer():
        while not stop_event.is_set() or not progress_queue.empty():
            try:
                msg = progress_queue.get(timeout=0.2)
                print(msg, flush=True)
            except Exception:
                pass
    printer = threading.Thread(target=_printer, daemon=True)
    printer.start()

    with ProcessPoolExecutor(max_workers=max_workers,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {executor.submit(_worker, t): t for t in tasks}
        for future in as_completed(futures):
            m_idx, a_idx, acc, history = future.result()
            results[m_idx, a_idx] = acc
            histories[m_idx][a_idx] = history
            method = active_method_keys[m_idx]
            attack = active_attack_keys[a_idx]
            done = int(pbar.n) + 1
            print(f"[{done}/{total}] {method.upper()} vs {attack} -> {acc:.2f}%", flush=True)
            pbar.set_postfix({'method': method, 'attack': attack, 'acc': f'{acc:.1f}%'})
            pbar.update(1)

    stop_event.set()
    printer.join(timeout=2)
    pbar.close()

    tag = f"{dataset_name}_byz{int(byz_ratio*100)}_alpha{noniid_alpha}"
    ds_upper = dataset_name.upper()

    # ── 热力图 ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(results, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)')
    ax.set_xticks(range(len(active_attack_keys)))
    ax.set_xticklabels(active_attack_labels, fontsize=11)
    ax.set_yticks(range(len(active_method_keys)))
    ax.set_yticklabels(active_method_labels, fontsize=11)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Aggregation Method', fontsize=12)
    ax.set_title(
        f'{ds_upper}: Accuracy (%) by Method × Attack\n'
        f'Byzantine={byz_ratio:.0%}, Dirichlet α={noniid_alpha}, Mesh degree=6',
        fontsize=12, fontweight='bold'
    )
    for m_idx in range(len(active_method_keys)):
        for a_idx in range(len(active_attack_keys)):
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
    for a_idx, (attack_key, attack_label) in enumerate(zip(active_attack_keys, active_attack_labels)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx, (method_label, color, ls) in enumerate(
                zip(active_method_labels, active_method_colors, active_method_linestyles)):
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
    for m_idx, (method_key, method_label) in enumerate(zip(active_method_keys, active_method_labels)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for a_idx, (attack_label, color, ls) in enumerate(
                zip(active_attack_labels,
                    [ATTACK_COLORS[i] for i in attack_indices],
                    [ATTACK_LINESTYLES[i] for i in attack_indices])):
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
    for label in active_attack_labels:
        print(f"  {label:>10}", end="")
    print()
    print("-" * 80)
    for m_idx, label in enumerate(active_method_labels):
        print(f"{label:>12}", end="")
        for a_idx in range(len(active_attack_keys)):
            print(f"  {results[m_idx, a_idx]:>9.2f}%", end="")
        print()

    # ── 原始数据保存（供二次绘图）────────────────────────────
    data = {
        'meta': {
            'dataset': dataset_name,
            'byzantine_ratio': byz_ratio,
            'noniid_alpha': noniid_alpha,
            'num_rounds': num_rounds,
            'log_interval': log_interval,
            'round_ticks': round_ticks,
            'methods': active_method_keys,
            'attacks': active_attack_keys,
        },
        'final_acc': {
            active_method_keys[m]: {active_attack_keys[a]: float(results[m, a])
                             for a in range(len(active_attack_keys))}
            for m in range(len(active_method_keys))
        },
        'histories': {
            active_method_keys[m]: {active_attack_keys[a]: histories[m][a]
                             for a in range(len(active_attack_keys))}
            for m in range(len(active_method_keys))
        },
    }
    import json
    json_path = save_dir / f'data_{tag}.json'
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved: {json_path.name}")

    # ── LaTeX 表格 ──────────────────────────────────────────
    print(f"\n% LaTeX table ({ds_upper}):")
    print("\\begin{table}[t]")
    print("\\centering")
    print(f"\\caption{{{ds_upper} Test Accuracy (\\%) under 6 Attack Types "
          f"(Byzantine={byz_ratio:.0%}, $\\alpha={noniid_alpha}$, Mesh $d=6$)}}")
    print(f"\\label{{tab:multi-attack-{dataset_name}}}")
    cols = "l" + "c" * len(active_attack_keys)
    print(f"\\begin{{tabular}}{{{cols}}}")
    print("\\toprule")
    print("Method & " + " & ".join(active_attack_labels) + " \\\\")
    print("\\midrule")
    best_per_col = results.argmax(axis=0)
    for m_idx, label in enumerate(active_method_labels):
        row_parts = []
        for a_idx in range(len(active_attack_keys)):
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
    base_config['federated']['num_workers'] = 2

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
    base_config['federated']['num_workers'] = 2

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
