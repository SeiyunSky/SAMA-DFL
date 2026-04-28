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
import yaml
import copy
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import SAMAAggregator, BALANCEAggregator, SCCLIPAggregator
from models import SimpleCNN
from utils import load_mnist, generate_ring_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack
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
            adaptive_alpha=config['sama'].get('adaptive_alpha', False)
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
    else:
        raise ValueError(f"Unknown method: {method}")


def train_single_run(config, method, device):
    """
    Run a single training experiment, return final accuracy.
    Simplified version of FederatedTrainer for sweep use.
    """
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
        num_workers=config['federated'].get('num_workers', 0)
    )

    # Topology
    neighbors = generate_ring_topology(num_clients)

    # Node split
    num_byzantine = int(num_clients * byz_ratio)
    honest_nodes = list(range(num_clients - num_byzantine))
    byzantine_nodes = list(range(num_clients - num_byzantine, num_clients))

    # Attack
    attack_type = config['attack']['type']
    if attack_type == 'gaussian':
        attack = GaussianAttack(std=config['attack']['gaussian_std'])
    elif attack_type == 'label_flipping':
        attack = LabelFlippingAttack(num_classes=10)
    elif attack_type == 'omniscient':
        attack = OmniscientAttack(amplification=config['attack'].get('amplification', 2.0))
    else:
        attack = None

    # Models and aggregator
    models = [SimpleCNN().to(device) for _ in range(num_clients)]
    aggregator = create_aggregator(method, config)

    # Training
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

        # Post-training attacks (Gaussian, Omniscient)
        if attack and not isinstance(attack, LabelFlippingAttack):
            if isinstance(attack, OmniscientAttack):
                honest_models = [local_models[i] for i in honest_nodes]
                for byz_id in byzantine_nodes:
                    local_models[byz_id] = attack.attack(honest_models)
            else:
                for byz_id in byzantine_nodes:
                    local_models[byz_id] = attack.attack(local_models[byz_id])

        # Aggregation
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

    # Final evaluation on honest mean
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
    methods = ['sama', 'balance', 'scclip']
    byz_ratios = [0.1, 0.2, 0.3, 0.4]
    results = {m: [] for m in methods}

    for byz_ratio in byz_ratios:
        for method in methods:
            config = copy.deepcopy(base_config)
            config['federated']['byzantine_ratio'] = byz_ratio
            # Use fewer rounds for sweep efficiency
            config['federated']['num_rounds'] = 150
            config['federated']['num_workers'] = 0

            print(f"\n  Byzantine={byz_ratio:.0%}, Method={method.upper()}...")
            acc = train_single_run(config, method, device)
            results[method].append(acc)
            print(f"  -> Accuracy: {acc:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    attack_type = base_config['attack']['type']
    noniid_alpha = base_config['data']['non_iid_alpha']
    fig.suptitle(f"Byzantine Robustness Curve | MNIST | Attack={attack_type} | α={noniid_alpha}",
                 fontsize=12, fontweight='bold')

    colors = {'sama': 'C0', 'balance': 'C1', 'scclip': 'C2'}
    markers = {'sama': 'o', 'balance': 's', 'scclip': '^'}

    byz_pcts = [r * 100 for r in byz_ratios]
    for method in methods:
        ax.plot(byz_pcts, results[method],
                label=method.upper(), color=colors[method],
                marker=markers[method], linewidth=2, markersize=8)

    ax.set_xlabel('Byzantine Ratio (%)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xticks(byz_pcts)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"byzantine_sweep_{attack_type}_alpha{noniid_alpha}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
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
    methods = ['sama', 'balance', 'scclip']
    alpha_values = [0.1, 0.3, 0.5, 1.0]
    results = {m: [] for m in methods}

    for alpha_val in alpha_values:
        for method in methods:
            config = copy.deepcopy(base_config)
            config['data']['non_iid_alpha'] = alpha_val
            config['federated']['num_rounds'] = 150
            config['federated']['num_workers'] = 0

            print(f"\n  Dirichlet α={alpha_val}, Method={method.upper()}...")
            acc = train_single_run(config, method, device)
            results[method].append(acc)
            print(f"  -> Accuracy: {acc:.2f}%")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    attack_type = base_config['attack']['type']
    byz_ratio = base_config['federated']['byzantine_ratio']
    fig.suptitle(f"Non-IID Sensitivity | MNIST | Attack={attack_type} | Byzantine={byz_ratio:.0%}",
                 fontsize=12, fontweight='bold')

    colors = {'sama': 'C0', 'balance': 'C1', 'scclip': 'C2'}
    markers = {'sama': 'o', 'balance': 's', 'scclip': '^'}

    for method in methods:
        ax.plot(alpha_values, results[method],
                label=method.upper(), color=colors[method],
                marker=markers[method], linewidth=2, markersize=8)

    ax.set_xlabel('Dirichlet α (higher = more IID)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_xticks(alpha_values)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"noniid_sweep_{attack_type}_byz{int(byz_ratio*100)}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
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
