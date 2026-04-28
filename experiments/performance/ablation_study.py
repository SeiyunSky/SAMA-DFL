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
import yaml
import copy
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import SAMAAggregator
from models import SimpleCNN
from utils import load_mnist, generate_ring_topology
from utils.topology import generate_mesh_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack
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
        num_workers=config['federated'].get('num_workers', 0)
    )

    topology_type = config['topology']['type']
    if topology_type == 'mesh':
        neighbors = generate_mesh_topology(num_clients, degree=config['topology']['degree'])
    else:
        neighbors = generate_ring_topology(num_clients)

    num_byzantine = int(num_clients * byz_ratio)
    honest_nodes = list(range(num_clients - num_byzantine))
    byzantine_nodes = list(range(num_clients - num_byzantine, num_clients))

    attack_type = config['attack']['type']
    if attack_type == 'gaussian':
        attack = GaussianAttack(std=config['attack']['gaussian_std'])
    elif attack_type == 'label_flipping':
        attack = LabelFlippingAttack(num_classes=10)
    elif attack_type == 'omniscient':
        attack = OmniscientAttack(amplification=config['attack'].get('amplification', 2.0))
    else:
        attack = None

    models = [SimpleCNN().to(device) for _ in range(num_clients)]

    for t in range(num_rounds):
        local_models = []
        for i in range(num_clients):
            model = models[i]
            if i in honest_nodes:
                model.train()
                for epoch in range(local_epochs):
                    for data, target in train_loaders[i]:
                        data, target = data.to(device), target.to(device)
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = torch.nn.functional.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
            elif attack and isinstance(attack, LabelFlippingAttack):
                model.train()
                for epoch in range(local_epochs):
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

        if attack and not isinstance(attack, LabelFlippingAttack):
            if isinstance(attack, OmniscientAttack):
                honest_models = [local_models[i] for i in honest_nodes]
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

    # Evaluate
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

    return 100.0 * correct / total


class NoAlignAggregator(SAMAAggregator):
    """SAMA without magnitude alignment — uses raw neighbor vectors for aggregation."""

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'num_filtered': 0, 'avg_trust': 0.0}
            return own_model

        w_i_vec = self.model_to_vector(own_model)
        w_i_norm = torch.norm(w_i_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_model, {'avg_trust': 0.0}
            return own_model

        neighbor_vecs = [self.model_to_vector(m) for m in neighbor_models]
        w_i_trust = self._extract_trust_vector(own_model)
        w_i_trust_norm = torch.norm(w_i_trust)
        neighbor_trust_vecs = [self._extract_trust_vector(m) for m in neighbor_models]

        trust_scores = []
        raw_vecs = []
        for idx, w_j_vec in enumerate(neighbor_vecs):
            w_j_norm = torch.norm(w_j_vec)
            if w_j_norm < self.eps:
                trust_scores.append(0.0)
                raw_vecs.append(None)
                continue
            w_j_trust = neighbor_trust_vecs[idx]
            w_j_trust_norm = torch.norm(w_j_trust)
            if w_j_trust_norm < self.eps or w_i_trust_norm < self.eps:
                trust_scores.append(0.0)
                raw_vecs.append(None)
                continue
            cos_sim = torch.dot(w_i_trust, w_j_trust) / (w_i_trust_norm * w_j_trust_norm)
            phi_j = max(0.0, cos_sim.item())
            trust_scores.append(phi_j)
            raw_vecs.append(w_j_vec if phi_j > 0 else None)

        valid_indices = [i for i, s in enumerate(trust_scores) if s > 0]
        if not valid_indices:
            if return_stats:
                return own_model, {'num_neighbors': len(neighbor_models), 'num_filtered': len(neighbor_models), 'avg_trust': 0.0}
            return own_model

        valid_scores = [trust_scores[i] for i in valid_indices]
        valid_vecs = [raw_vecs[i] for i in valid_indices]
        total_weight = sum(valid_scores) + self.eps
        agg_vec = sum(s * v for s, v in zip(valid_scores, valid_vecs)) / total_weight
        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            return agg_model, {'num_neighbors': len(neighbor_models),
                              'num_filtered': len(neighbor_models) - len(valid_indices),
                              'avg_trust': np.mean(valid_scores)}
        return agg_model


class NoDirectionAggregator(SAMAAggregator):
    """SAMA without direction filtering — accepts all neighbors equally."""

    def aggregate(self, own_model, neighbor_models, t=0, T=100, return_stats=False):
        if not neighbor_models:
            if return_stats:
                return own_model, {'num_neighbors': 0, 'avg_trust': 1.0}
            return own_model

        w_i_vec = self.model_to_vector(own_model)
        w_i_norm = torch.norm(w_i_vec)
        if w_i_norm < self.eps:
            if return_stats:
                return own_model, {'avg_trust': 1.0}
            return own_model

        # Magnitude alignment but equal weights (no cos filtering)
        aligned_vecs = []
        for m in neighbor_models:
            w_j_vec = self.model_to_vector(m)
            w_j_norm = torch.norm(w_j_vec)
            if w_j_norm < self.eps:
                continue
            aligned = w_i_norm * (w_j_vec / w_j_norm)
            aligned_vecs.append(aligned)

        if not aligned_vecs:
            if return_stats:
                return own_model, {'num_neighbors': len(neighbor_models), 'avg_trust': 1.0}
            return own_model

        agg_vec = torch.stack(aligned_vecs).mean(dim=0)
        agg_model = self.vector_to_model(agg_vec, own_model)

        if return_stats:
            return agg_model, {'num_neighbors': len(neighbor_models),
                              'num_filtered': 0, 'avg_trust': 1.0}
        return agg_model


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

    # Define ablation variants
    variants = {
        'Full SAMA': SAMAAggregator(
            alpha=sama_cfg['alpha'],
            trust_layers=sama_cfg.get('trust_layers', None),
        ),
        'No trust_layers': SAMAAggregator(
            alpha=sama_cfg['alpha'],
            trust_layers=None,  # Use full model for cos similarity
        ),
        'No alignment': NoAlignAggregator(
            alpha=sama_cfg['alpha'],
            trust_layers=sama_cfg.get('trust_layers', None),
        ),
        'No direction': NoDirectionAggregator(
            alpha=sama_cfg['alpha'],
        ),
        'No self-anchor': SAMAAggregator(
            alpha=0.0,  # alpha=0 means fully trust aggregation
            trust_layers=sama_cfg.get('trust_layers', None),
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

    attack_type = config['attack']['type']
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
