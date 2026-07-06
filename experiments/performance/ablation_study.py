"""
Ablation Study for SAMA-DFL
Test each component's contribution by disabling it one at a time.
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

sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import SAMAAggregator
from models import SimpleCNN
from utils import load_mnist, load_mnist_gpu, generate_ring_topology
from utils import make_run_log_dir, open_task_log, append_task_log, finalize_task_log
from utils.topology import generate_mesh_topology
from attacks import (GaussianAttack, LabelFlippingAttack, OmniscientAttack,
                     KrumAttack, TrimAttack, BrokenNodeAttack, NoAttack)
from collections import OrderedDict


def train_ablation_run(config, aggregator, device, task_log=None):
    """Train with a specific aggregator, return (final_accuracy, acc_history)."""
    num_clients = config['federated']['num_clients']
    byz_ratio = config['federated']['byzantine_ratio']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['optimizer']['lr']
    log_interval = config['logging']['log_interval']
    use_gpu_loader = bool(int(os.getenv('GPU_LOADER', '1')))
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
            num_workers=config['federated'].get('num_workers', 2)
        )

    topology_type = config['topology']['type']
    if topology_type == 'mesh':
        neighbors = generate_mesh_topology(num_clients, degree=config['topology']['degree'])
    else:
        neighbors = generate_ring_topology(num_clients)

    num_byzantine = int(num_clients * byz_ratio)
    honest_nodes = set(range(num_clients - num_byzantine))
    byzantine_nodes = set(range(num_clients - num_byzantine, num_clients))

    attack_type = os.getenv('ATTACK_TYPE', config['attack']['type']).lower()
    num_byzantine = int(num_clients * byz_ratio)
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
    elif attack_type == 'broken_node':
        attack = BrokenNodeAttack()
    else:
        attack = NoAttack()

    models = [SimpleCNN().to(device) for _ in range(num_clients)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in models]
    acc_history = []

    for t in range(num_rounds):
        local_vecs = [None] * num_clients
        for i in range(num_clients):
            model = models[i]
            is_training = (i in honest_nodes) or isinstance(attack, NoAttack)
            if is_training:
                model.train()
                for epoch in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                        optimizer = optimizers[i]
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
                        optimizer = optimizers[i]
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            local_vecs[i] = aggregator.model_to_vector(models[i])

        no_attack = isinstance(attack, (NoAttack, BrokenNodeAttack))
        if not no_attack and not isinstance(attack, LabelFlippingAttack):
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

        # 按 log_interval 评估
        if (t + 1) % log_interval == 0:
            honest_vecs_eval = [updated_vecs[i] for i in honest_nodes]
            eval_mean = torch.stack(honest_vecs_eval).mean(dim=0)
            _eval_m = SimpleCNN().to(device)
            aggregator.load_from_vector(_eval_m, eval_mean)
            _eval_m.eval()
            c_, n_ = 0, 0
            with torch.no_grad():
                for d_, t_ in test_loader:
                    d_, t_ = d_.to(device, non_blocking=True), t_.to(device, non_blocking=True)
                    c_ += _eval_m(d_).argmax(dim=1).eq(t_).sum().item()
                    n_ += t_.size(0)
            round_acc = 100.0 * c_ / n_
            acc_history.append(round_acc)
            if task_log is not None:
                append_task_log(task_log, t + 1, num_rounds, round_acc)
            del _eval_m

    # Evaluate
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
    del train_loaders, test_loader, models, optimizers, global_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return acc, acc_history


class NoAlignAggregator(SAMAAggregator):
    """SAMA without magnitude alignment — uses raw neighbor vectors for aggregation."""

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'num_filtered': 0, 'avg_trust': 0.0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        w_i_norm = torch.norm(own_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': N, 'avg_trust': 0.0}
            return own_vec

        w_i_trust = self._extract_trust_vec(own_vec)
        w_i_trust_norm = torch.norm(w_i_trust)

        if self._trust_slices is None:
            neighbor_trust = neighbor_mat
        else:
            neighbor_trust = torch.stack([self._extract_trust_vec(neighbor_mat[i]) for i in range(N)])
        neighbor_trust_norms = torch.norm(neighbor_trust, dim=1)
        neighbor_norms = torch.norm(neighbor_mat, dim=1)

        valid_mask = (neighbor_norms >= self.eps) & (neighbor_trust_norms >= self.eps) & (w_i_trust_norm >= self.eps)
        cos_sims = torch.zeros(N, device=own_vec.device)
        if valid_mask.any():
            dots = (neighbor_trust[valid_mask] * w_i_trust).sum(dim=1)
            cos_sims[valid_mask] = dots / (neighbor_trust_norms[valid_mask] * w_i_trust_norm)

        phi = torch.clamp(cos_sims, min=0.0)
        valid = (phi > 0) & valid_mask

        if not valid.any():
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': N, 'avg_trust': 0.0}
            return own_vec

        # 无幅度对齐：直接用原始邻居向量加权
        valid_vecs = neighbor_mat[valid]
        valid_phi = phi[valid]
        total_weight = valid_phi.sum() + self.eps
        agg_vec = (valid_phi.unsqueeze(1) * valid_vecs).sum(dim=0) / total_weight

        if return_stats:
            return agg_vec, {'num_neighbors': N,
                             'num_filtered': N - int(valid.sum()),
                             'avg_trust': valid_phi.mean().item()}
        return agg_vec


class NoDirectionAggregator(SAMAAggregator):
    """SAMA without direction filtering — accepts all neighbors equally."""

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'num_filtered': 0, 'avg_trust': 1.0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        w_i_norm = torch.norm(own_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': 0, 'avg_trust': 1.0}
            return own_vec

        neighbor_norms = torch.norm(neighbor_mat, dim=1)
        valid_mask = neighbor_norms >= self.eps

        if not valid_mask.any():
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': N, 'avg_trust': 1.0}
            return own_vec

        # 幅度对齐，等权（无方向过滤）
        valid_vecs = neighbor_mat[valid_mask]
        valid_norms = neighbor_norms[valid_mask]
        aligned = w_i_norm * (valid_vecs / valid_norms.unsqueeze(1))
        agg_vec = aligned.mean(dim=0)

        if return_stats:
            return agg_vec, {'num_neighbors': N,
                             'num_filtered': 0,
                             'avg_trust': 1.0}
        return agg_vec


class HardThresholdAggregator(SAMAAggregator):
    """
    SAMA 硬阈值变体：方向余弦 > 0 的邻居全部接受（均等权重），不使用软加权。
    对比软加权（φ_j = cos_sim），验证连续权重 vs 二值权重的差异。
    """

    def aggregate(self, own_vec, neighbor_vecs, t=0, T=100, return_stats=False, **kwargs):
        if isinstance(neighbor_vecs, list):
            if not neighbor_vecs:
                if return_stats:
                    return own_vec, {'num_neighbors': 0, 'num_filtered': 0, 'avg_trust': 0.0}
                return own_vec
            neighbor_mat = torch.stack(neighbor_vecs)
        else:
            neighbor_mat = neighbor_vecs

        N = neighbor_mat.shape[0]
        w_i_norm = torch.norm(own_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': N, 'avg_trust': 0.0}
            return own_vec

        w_i_trust = self._extract_trust_vec(own_vec)
        w_i_trust_norm = torch.norm(w_i_trust)

        if self._trust_slices is None:
            neighbor_trust = neighbor_mat
        else:
            neighbor_trust = torch.stack([self._extract_trust_vec(neighbor_mat[i]) for i in range(N)])
        neighbor_trust_norms = torch.norm(neighbor_trust, dim=1)
        neighbor_norms = torch.norm(neighbor_mat, dim=1)

        valid_mask = (neighbor_norms >= self.eps) & (neighbor_trust_norms >= self.eps) & (w_i_trust_norm >= self.eps)
        cos_sims = torch.zeros(N, device=own_vec.device)
        if valid_mask.any():
            dots = (neighbor_trust[valid_mask] * w_i_trust).sum(dim=1)
            cos_sims[valid_mask] = dots / (neighbor_trust_norms[valid_mask] * w_i_trust_norm)

        # 硬阈值：cos > 0 则接受，均等权重
        accept = (cos_sims > 0) & valid_mask
        if not accept.any():
            if return_stats:
                return own_vec, {'num_neighbors': N, 'num_filtered': N, 'avg_trust': 0.0}
            return own_vec

        valid_vecs = neighbor_mat[accept]
        valid_norms = neighbor_norms[accept]
        aligned = w_i_norm * (valid_vecs / valid_norms.unsqueeze(1))
        agg_vec = aligned.mean(dim=0)

        num_accepted = int(accept.sum())
        if return_stats:
            return agg_vec, {'num_neighbors': N,
                             'num_filtered': N - num_accepted,
                             'avg_trust': num_accepted / N}
        return agg_vec


def _build_ablation_aggregator(spec, config, device):
    """在子进程里根据 spec 重建 aggregator（避免跨进程传递含 CUDA tensor 的对象）。"""
    sama_cfg = config['sama']
    template = SimpleCNN().to(device)
    kind = spec['kind']
    if kind == 'sama':
        return SAMAAggregator(
            alpha=spec['alpha'],
            trust_layers=spec['trust_layers'],
            model_template=template,
        )
    elif kind == 'no_align':
        return NoAlignAggregator(
            alpha=spec['alpha'],
            trust_layers=spec['trust_layers'],
            model_template=template,
        )
    elif kind == 'no_direction':
        return NoDirectionAggregator(
            alpha=spec['alpha'],
            model_template=template,
        )
    elif kind == 'hard_threshold':
        return HardThresholdAggregator(
            alpha=spec['alpha'],
            trust_layers=spec['trust_layers'],
            model_template=template,
        )
    else:
        raise ValueError(f"Unknown ablation kind: {kind}")


def _ablation_worker(args):
    """顶层 worker，可被 ProcessPoolExecutor pickle。"""
    v_idx, name, spec, config, device_str, progress_queue, run_log_dir, run_meta_base = args

    seed = config.get('experiment', {}).get('seed', 42)
    torch.manual_seed(seed + v_idx * 17)
    np.random.seed(seed + v_idx * 17)

    import time
    device = torch.device(device_str)
    config = copy.deepcopy(config)
    config['federated']['num_workers'] = 0  # 子进程里禁用 DataLoader workers

    aggregator = _build_ablation_aggregator(spec, config, device)

    _meta = dict(run_meta_base)
    _meta['variant'] = name
    log_dir = Path(run_log_dir)
    attack_key = _meta.get('attack', 'unknown')
    log_filename = f"{name.lower().replace(' ', '_')}_{attack_key}.log"
    task_log = open_task_log(
        log_dir,
        log_filename,
        _meta,
    )

    if progress_queue is not None:
        progress_queue.put(f"[{name}] started")

    t0 = time.time()
    acc, history = train_ablation_run(config, aggregator, device, task_log=task_log)
    elapsed = time.time() - t0
    finalize_task_log(task_log, acc, elapsed)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if progress_queue is not None:
        progress_queue.put(f"[{name}] done accuracy={acc:.2f}%")

    return v_idx, name, acc, elapsed, history


def run_ablation_study(config_path=None):
    """
    Ablation study: disable each SAMA component one at a time.
    """
    print("=" * 80)
    print("Ablation Study: SAMA-DFL Component Analysis")
    print("=" * 80)

    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config['experiment']['device'])
    sama_cfg = config['sama']

    # variant 用纯数据描述，子进程里重建（避免跨进程传 CUDA 对象）
    variant_specs = [
        ('Full SAMA',      {'kind': 'sama',          'alpha': sama_cfg['alpha'],  'trust_layers': sama_cfg.get('trust_layers', None)}),
        ('No alignment',   {'kind': 'no_align',      'alpha': sama_cfg['alpha'],  'trust_layers': sama_cfg.get('trust_layers', None)}),
        ('No direction',   {'kind': 'no_direction',  'alpha': sama_cfg['alpha'],  'trust_layers': None}),
        ('No self-anchor', {'kind': 'sama',          'alpha': 0.0,                'trust_layers': sama_cfg.get('trust_layers', None)}),
        ('Hard threshold', {'kind': 'hard_threshold','alpha': sama_cfg['alpha'],  'trust_layers': sama_cfg.get('trust_layers', None)}),
    ]

    # 支持多攻击：ABLATION_ATTACKS=gaussian,omniscient 或留空=全部
    _attacks_env = os.getenv('ABLATION_ATTACKS', '').strip()
    VALID_ATTACKS = ['none', 'gaussian', 'label_flipping', 'omniscient',
                     'krum_attack', 'trim_attack', 'broken_node']
    if _attacks_env:
        attack_keys = [k.strip() for k in _attacks_env.split(',') if k.strip() in VALID_ATTACKS]
        if not attack_keys:
            attack_keys = list(VALID_ATTACKS)
    else:
        # 未设置 ABLATION_ATTACKS：跑全部（忽略 ATTACK_TYPE，避免单攻击污染）
        attack_keys = list(VALID_ATTACKS)

    byz_ratio = config['federated']['byzantine_ratio']
    noniid_alpha = config['data']['non_iid_alpha']

    from datetime import datetime
    run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = make_run_log_dir(
        'ablation',
        run_ts,
        base_dir=Path(__file__).parent.parent.parent / 'results',
    )
    print(f"  Run logs → {run_log_dir}", flush=True)
    print(f"  Attacks: {attack_keys}", flush=True)

    max_workers = int(os.getenv('TABLE_WORKERS',
                                config.get('experiment', {}).get('parallel_workers', 4)))
    device_str = str(device)
    name_order = [name for name, _ in variant_specs]

    # 每个 attack 独立跑一轮所有 variant
    all_results = {}    # attack_key -> {name: acc}
    all_histories = {}  # attack_key -> {name: history}

    for attack_key in attack_keys:
        print(f"\n{'='*60}", flush=True)
        print(f"  Attack: {attack_key}", flush=True)

        run_meta_base = {
            'exp_name'    : 'ablation_study',
            'dataset'     : 'mnist',
            'attack'      : attack_key,
            'byz_ratio'   : f"{byz_ratio:.2f}",
            'noniid_alpha': str(noniid_alpha),
            'num_clients' : str(config['federated']['num_clients']),
            'num_rounds'  : str(config['federated']['num_rounds']),
            'lr'          : str(config['optimizer']['lr']),
            'batch_size'  : str(config['federated']['batch_size']),
        }

        # 每个 task 用独立的 config copy，写入当前 attack
        tasks = []
        for v_idx, (name, spec) in enumerate(variant_specs):
            cfg = copy.deepcopy(config)
            cfg['attack']['type'] = attack_key
            tasks.append((v_idx, name, spec, cfg, device_str, None,
                          str(run_log_dir), run_meta_base))
        total = len(tasks)

        mgr = multiprocessing.Manager()
        progress_queue = mgr.Queue()
        tasks = [
            (v_idx, name, spec, cfg, dev, progress_queue, log_dir, meta)
            for v_idx, name, spec, cfg, dev, _, log_dir, meta in tasks
        ]

        done_count = [0]
        stop_event = threading.Event()
        def _printer():
            while not stop_event.is_set() or not progress_queue.empty():
                try:
                    msg = progress_queue.get(timeout=0.2)
                    if 'done' in msg:
                        done_count[0] += 1
                        print(f"  [{done_count[0]}/{total}] {msg}", flush=True)
                    else:
                        print(f"  {msg}", flush=True)
                except Exception:
                    pass
        printer = threading.Thread(target=_printer, daemon=True)
        printer.start()

        results = {}
        elapsed_map = {}
        histories = {}

        with ProcessPoolExecutor(max_workers=max_workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = {executor.submit(_ablation_worker, t): t for t in tasks}
            for future in as_completed(futures):
                v_idx, name, acc, elapsed, history = future.result()
                results[name] = acc
                elapsed_map[name] = elapsed
                histories[name] = history

        stop_event.set()
        printer.join(timeout=3)

        for name in name_order:
            print(f"  -> {name}: {results[name]:.2f}%  ({elapsed_map[name]:.1f}s)")

        all_results[attack_key] = results
        all_histories[attack_key] = histories

        # ── Bar chart（每个 attack 一张）────────────────────
        save_dir = Path(__file__).parent.parent.parent / 'results'
        save_dir.mkdir(exist_ok=True)
        tag = f"ablation_{attack_key}_byz{int(byz_ratio * 100)}"

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(
            f"Ablation Study | Attack={attack_key} | Byzantine={byz_ratio:.0%} | α={noniid_alpha}",
            fontsize=12, fontweight='bold'
        )
        names = name_order
        accs = [results[n] for n in names]
        colors = ['#2196F3' if n == 'Full SAMA' else '#FF9800' for n in names]
        bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        full_acc = results['Full SAMA']
        for i, name in enumerate(name_order):
            acc = results[name]
            if name != 'Full SAMA':
                drop = full_acc - acc
                if drop > 0:
                    ax.text(i, acc / 2, f'-{drop:.1f}pp', ha='center', va='center',
                           fontsize=10, color='red', fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / f"{tag}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Bar chart saved: {tag}.png")

        # ── 收敛曲线图（每个 attack 一张，6 variant 叠在一起）──
        log_interval = config['logging']['log_interval']
        num_rounds_cfg = config['federated']['num_rounds']
        round_ticks = list(range(log_interval, num_rounds_cfg + 1, log_interval))
        variant_colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0', '#795548']
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for color, name in zip(variant_colors, name_order):
            hist = histories.get(name, [])
            if hist:
                ax2.plot(round_ticks[:len(hist)], hist,
                         label=name, color=color,
                         linewidth=1.8, marker='o', markersize=3)
        ax2.set_xlabel('Communication Round', fontsize=12)
        ax2.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax2.set_title(
            f"Ablation Convergence | Attack={attack_key} | Byzantine={byz_ratio:.0%} | α={noniid_alpha}",
            fontsize=12, fontweight='bold'
        )
        ax2.set_ylim(0, 100)
        ax2.legend(fontsize=9, ncol=2, loc='lower right')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / f"{tag}_convergence.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Convergence curve saved: {tag}_convergence.png")

        # ── JSON ─────────────────────────────────────────────
        import json
        json_path = save_dir / f"{tag}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'meta': {
                    'attack_type': attack_key,
                    'byz_ratio': byz_ratio,
                    'noniid_alpha': noniid_alpha,
                    'num_rounds': num_rounds_cfg,
                    'log_interval': log_interval,
                    'round_ticks': round_ticks,
                },
                'results': results,
                'histories': {name: histories.get(name, []) for name in name_order},
            }, f, indent=2)
        print(f"  Raw data saved: {json_path.name}")

        # ── 控制台表格 ────────────────────────────────────────
        print(f"\n  {'Variant':>20}  {'Accuracy':>10}  {'Drop':>8}")
        print(f"  {'-'*48}")
        for name in name_order:
            acc = results[name]
            drop = full_acc - acc
            drop_str = f"-{drop:.1f}pp" if name != 'Full SAMA' else "-"
            print(f"  {name:>20}  {acc:>9.2f}%  {drop_str:>8}")

    return all_results


if __name__ == "__main__":
    run_ablation_study()
