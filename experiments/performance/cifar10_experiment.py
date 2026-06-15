"""
CIFAR-10 Full Training Experiment
Compare SAMA-DFL and BALANCE performance on CIFAR-10
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
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

plt.rcParams['font.family'] = 'DejaVu Sans'

from aggregators import SAMAAggregator, BALANCEAggregator, SCCLIPAggregator
from models import SimpleCNN
from utils import load_cifar10, generate_ring_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack, KrumAttack, TrimAttack
from collections import OrderedDict


class CIFAR10Trainer:
    """CIFAR-10训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['experiment']['device'])

        # 数据加载 - vGPU优化
        num_workers = config.get('federated', {}).get('num_workers', 8)
        self.train_loaders, self.test_loader = load_cifar10(
            data_dir=config['data']['data_dir'],
            num_clients=config['federated']['num_clients'],
            alpha=config['data']['non_iid_alpha'],
            batch_size=config['federated']['batch_size'],
            num_workers=num_workers
        )

        # 网络拓扑
        self.num_clients = config['federated']['num_clients']
        self.neighbors = generate_ring_topology(self.num_clients)

        # 节点划分
        num_byzantine = int(self.num_clients * config['federated']['byzantine_ratio'])
        self.honest_nodes = list(range(self.num_clients - num_byzantine))
        self.byzantine_nodes = list(range(self.num_clients - num_byzantine, self.num_clients))

        # 攻击（环境变量 ATTACK_TYPE 可覆盖配置文件）
        attack_type = os.getenv('ATTACK_TYPE', config['attack']['type'])
        if attack_type == 'gaussian':
            self.attack = GaussianAttack(std=config['attack']['gaussian_std'])
        elif attack_type == 'label_flipping':
            self.attack = LabelFlippingAttack(num_classes=10)
        elif attack_type == 'omniscient':
            self.attack = OmniscientAttack(amplification=config['attack'].get('amplification', 2.0))
        elif attack_type == 'krum_attack':
            self.attack = KrumAttack(num_byzantine=num_byzantine,
                                     amplification=config['attack'].get('amplification', 1.0))
        elif attack_type == 'trim_attack':
            self.attack = TrimAttack(num_byzantine=num_byzantine,
                                     trim_ratio=config['attack'].get('trim_ratio', 0.1))
        else:
            self.attack = None

    def train(self, method='sama'):
        """Training main loop"""
        print(f"\nTraining {method.upper()} on CIFAR-10...")

        # 初始化模型（CIFAR-10使用更深的CNN）
        from models import SimpleCNN
        models = [SimpleCNN(num_classes=10, in_channels=3).to(self.device)
                 for _ in range(self.num_clients)]

        # 聚合器
        if method == 'sama':
            aggregator = SAMAAggregator(
                alpha=self.config['sama']['alpha'],
                use_temperature=self.config['sama'].get('use_temperature', False),
                trust_layers=self.config['sama'].get('trust_layers', None),
            )
        elif method == 'scclip':
            aggregator = SCCLIPAggregator(
                alpha=self.config['scclip']['alpha'],
                clip_constant=self.config['scclip']['clip_constant']
            )
        else:
            aggregator = BALANCEAggregator(
                alpha=self.config['balance']['alpha'],
                gamma=self.config['balance']['gamma'],
                kappa=self.config['balance']['kappa']
            )

        history = {'test_loss': [], 'test_acc': [], 'consensus_error': []}

        num_rounds = self.config['federated']['num_rounds']
        local_epochs = self.config['federated']['local_epochs']
        lr = self.config['optimizer']['lr']

        pbar = tqdm(range(num_rounds), desc=f"{method.upper()}")
        for t in pbar:
            # 本地训练
            local_models = []
            for i in range(self.num_clients):
                model = models[i]

                if i in self.honest_nodes:
                    model.train()
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                                momentum=self.config['optimizer'].get('momentum', 0.0),
                                                weight_decay=self.config['optimizer'].get('weight_decay', 0.0))
                    for epoch in range(local_epochs):
                        for data, target in self.train_loaders[i]:
                            data, target = data.to(self.device), target.to(self.device)
                            optimizer.zero_grad()
                            output = model(data)
                            loss = torch.nn.functional.cross_entropy(output, target)
                            loss.backward()
                            optimizer.step()
                elif self.attack and isinstance(self.attack, LabelFlippingAttack):
                    model.train()
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                                momentum=self.config['optimizer'].get('momentum', 0.0),
                                                weight_decay=self.config['optimizer'].get('weight_decay', 0.0))
                    for epoch in range(local_epochs):
                        for data, target in self.train_loaders[i]:
                            data, target = data.to(self.device), target.to(self.device)
                            target = self.attack.flip_labels(target)
                            optimizer.zero_grad()
                            output = model(data)
                            loss = torch.nn.functional.cross_entropy(output, target)
                            loss.backward()
                            optimizer.step()

                local_models.append(model.state_dict())

            # 拜占庭攻击（非Label Flipping类型）
            if self.attack and not isinstance(self.attack, LabelFlippingAttack):
                honest_models = [local_models[i] for i in self.honest_nodes]
                if isinstance(self.attack, (OmniscientAttack, KrumAttack, TrimAttack)):
                    for byz_id in self.byzantine_nodes:
                        local_models[byz_id] = self.attack.attack(honest_models)
                else:
                    for byz_id in self.byzantine_nodes:
                        local_models[byz_id] = self.attack.attack(local_models[byz_id])

            # 聚合
            updated_models = []
            for i in range(self.num_clients):
                own_model = local_models[i]
                neighbor_models = [local_models[j] for j in self.neighbors[i]]

                if i in self.honest_nodes:
                    aggregated, agg_stats = aggregator.aggregate(
                        own_model, neighbor_models, t=t, T=num_rounds, return_stats=True
                    )
                    avg_trust = agg_stats.get('avg_trust', None)
                    final = aggregator.final_update(own_model, aggregated, avg_trust=avg_trust)
                else:
                    final = own_model

                updated_models.append(final)

            models = [SimpleCNN(num_classes=10, in_channels=3).to(self.device)
                     for _ in range(self.num_clients)]
            for i, state_dict in enumerate(updated_models):
                models[i].load_state_dict(state_dict)

            # 评估
            if (t + 1) % self.config['logging']['log_interval'] == 0:
                # 共识误差
                honest_vecs = [aggregator.model_to_vector(updated_models[i])
                              for i in self.honest_nodes]
                honest_vecs = torch.stack(honest_vecs)
                honest_mean = honest_vecs.mean(dim=0)
                D_t = torch.mean(torch.norm(honest_vecs - honest_mean, dim=1).pow(2)).item()

                # 测试
                global_model = SimpleCNN(num_classes=10, in_channels=3).to(self.device)
                global_model.load_state_dict(
                    aggregator.vector_to_model(honest_mean, global_model.state_dict())
                )
                global_model.eval()

                correct, total, loss_sum = 0, 0, 0
                with torch.no_grad():
                    for data, target in self.test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = global_model(data)
                        loss_sum += torch.nn.functional.cross_entropy(
                            output, target, reduction='sum'
                        ).item()
                        pred = output.argmax(dim=1)
                        correct += pred.eq(target).sum().item()
                        total += target.size(0)

                test_loss = loss_sum / total
                test_acc = 100.0 * correct / total

                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)
                history['consensus_error'].append(D_t)

                pbar.set_postfix({'loss': f'{test_loss:.4f}', 'acc': f'{test_acc:.2f}%'})

        return history


def run_cifar10_experiment(config_path=None):
    """Run CIFAR-10 experiment"""
    print("=" * 80)
    print("Experiment C2: CIFAR-10 Full Training")
    print("=" * 80)

    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'cifar10.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    trainer = CIFAR10Trainer(config)

    # 训练
    methods = ['sama', 'balance', 'scclip']
    results = {}

    for method in methods:
        results[method] = trainer.train(method=method)

    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    attack_type = config['attack']['type']
    byz_ratio = config['federated']['byzantine_ratio']
    noniid_alpha = config['data']['non_iid_alpha']
    fig.suptitle(f"CIFAR-10 | Attack={attack_type} | Byzantine={byz_ratio:.0%} | Dirichlet α={noniid_alpha}",
                 fontsize=13, fontweight='bold')

    log_interval = config['logging']['log_interval']
    colors = {'sama': 'C0', 'balance': 'C1', 'scclip': 'C2'}

    # Test loss
    for method in methods:
        t_vals = np.arange(log_interval, config['federated']['num_rounds'] + 1, log_interval)
        axes[0].plot(t_vals, results[method]['test_loss'],
                    label=method.upper(), color=colors[method], linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Training Round')
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('CIFAR-10: Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Test accuracy
    for method in methods:
        t_vals = np.arange(log_interval, config['federated']['num_rounds'] + 1, log_interval)
        axes[1].plot(t_vals, results[method]['test_acc'],
                    label=method.upper(), color=colors[method], linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Training Round')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('CIFAR-10: Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Final performance comparison
    final_accs = [results[method]['test_acc'][-1] for method in methods]
    x = np.arange(len(methods))
    bars = axes[2].bar(x, final_accs, color=[colors[m] for m in methods], alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([m.upper() for m in methods])
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].set_title('Final Performance Comparison')
    axes[2].grid(True, alpha=0.3, axis='y')

    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"cifar10_{attack_type}_byz{int(byz_ratio*100)}_alpha{noniid_alpha}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_dir / fname}")

    # Print results
    print("\n" + "=" * 80)
    print("CIFAR-10 Final Results")
    print("=" * 80)
    for method in methods:
        final_loss = results[method]['test_loss'][-1]
        final_acc = results[method]['test_acc'][-1]
        print(f"{method.upper():8s}: Loss={final_loss:.4f}, Acc={final_acc:.2f}%")

    return results


if __name__ == "__main__":
    results = run_cifar10_experiment()
