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
from utils import load_mnist, generate_ring_topology
from utils.topology import generate_mesh_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack, KrumAttack, TrimAttack
from collections import OrderedDict


def train_ablation_run(config, aggregator, device):
    """Train with a specific aggregator, return final accuracy."""
    num_clients = config['federated']['num_clients']
    byz_ratio = config['federated']['byzantine_ratio']
    num_rounds = config['federated']['num_rounds']
    local_epochs = config['federated']['local_epochs']
    lr = config['optimizer']['lr']

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

    attack_type = os.getenv('ATTACK_TYPE', config['attack']['type'])
    if attack_type == 'gaussian':
        attack = GaussianAttack(std=config['attack']['gaussian_std'])
    elif attack_type == 'label_flipping':
        attack = LabelFlippingAttack(num_classes=10)
    elif attack_type == 'omniscient':
        attack = OmniscientAttack(amplification=config['attack'].get('amplification', 2.0))
    else:
        attack = None

    models = [SimpleCNN().to(device) for _ in range(num_clients)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in models]

    for t in range(num_rounds):
        local_vecs = [None] * num_clients
        for i in range(num_clients):
            model = models[i]
            if i in honest_nodes:
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

        if attack and not isinstance(attack, LabelFlippingAttack):
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

    return 100.0 * correct / total


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
    _template_model = SimpleCNN().to(device)

    # Define ablation variants
    variants = {
        'Full SAMA': SAMAAggregator(
            alpha=sama_cfg['alpha'],
            trust_layers=sama_cfg.get('trust_layers', None),
            model_template=_template_model,
        ),
        'No trust_layers': SAMAAggregator(
            alpha=sama_cfg['alpha'],
            trust_layers=None,
            model_template=_template_model,
        ),
        'No alignment': NoAlignAggregator(
            alpha=sama_cfg['alpha'],
            trust_layers=sama_cfg.get('trust_layers', None),
            model_template=_template_model,
        ),
        'No direction': NoDirectionAggregator(
            alpha=sama_cfg['alpha'],
            model_template=_template_model,
        ),
        'No self-anchor': SAMAAggregator(
            alpha=0.0,
            trust_layers=sama_cfg.get('trust_layers', None),
            model_template=_template_model,
        ),
        'Hard threshold': HardThresholdAggregator(
            alpha=sama_cfg['alpha'],
            trust_layers=sama_cfg.get('trust_layers', None),
            model_template=_template_model,
        ),
    }

    results = {}
    for name, aggregator in variants.items():
        print(f"\n  Running: {name}...")
        acc = train_ablation_run(config, aggregator, device)
        results[name] = acc
        print(f"  -> Accuracy: {acc:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    attack_type = os.getenv('ATTACK_TYPE', config['attack']['type'])
    byz_ratio = config['federated']['byzantine_ratio']
    noniid_alpha = config['data']['non_iid_alpha']
    fig.suptitle(f"Ablation Study | Attack={attack_type} | Byzantine={byz_ratio:.0%} | α={noniid_alpha}",
                 fontsize=12, fontweight='bold')

    names = list(results.keys())
    accs = list(results.values())
    colors = ['#2196F3' if n == 'Full SAMA' else '#FF9800' for n in names]

    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add drop annotations
    full_acc = results['Full SAMA']
    for i, (name, acc) in enumerate(results.items()):
        if name != 'Full SAMA':
            drop = full_acc - acc
            if drop > 0:
                ax.text(i, acc / 2, f'-{drop:.1f}pp', ha='center', va='center',
                       fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"ablation_{attack_type}_byz{int(byz_ratio * 100)}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    plt.close()
    import json
    json_path = save_dir / fname.replace('.png', '.json')
    with open(json_path, 'w') as f:
        json.dump({
            'meta': {'attack_type': attack_type, 'byz_ratio': byz_ratio},
            'results': results,
        }, f, indent=2)
    print(f"Raw data saved: {json_path.name}")
    print(f"\nPlot saved to: {save_dir / fname}")

    # Print table
    print("\n" + "=" * 50)
    print(f"{'Variant':>20}  {'Accuracy':>10}  {'Drop':>8}")
    print("-" * 50)
    for name, acc in results.items():
        drop = full_acc - acc
        drop_str = f"-{drop:.1f}pp" if name != 'Full SAMA' else "-"
        print(f"{name:>20}  {acc:>9.2f}%  {drop_str:>8}")

    return results


if __name__ == "__main__":
    run_ablation_study()
