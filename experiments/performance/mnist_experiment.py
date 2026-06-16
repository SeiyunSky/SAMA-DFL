"""
MNIST Full Training Experiment
Compare SAMA-DFL, BALANCE, SC-CLIP performance on MNIST
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

from aggregators import (SAMAAggregator, BALANCEAggregator, SCCLIPAggregator,
                         FedAvgAggregator, KrumAggregator, TrimmedMeanAggregator,
                         CoordMedianAggregator)
from models import SimpleCNN
from utils import load_mnist, generate_ring_topology
from attacks import GaussianAttack, LabelFlippingAttack, OmniscientAttack, KrumAttack, TrimAttack
from collections import OrderedDict


class FederatedTrainer:
    """联邦学习训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['experiment']['device'])

        # 数据加载 - vGPU优化
        num_workers = config.get('federated', {}).get('num_workers', 8)
        self.train_loaders, self.test_loader = load_mnist(
            data_dir=config['data']['data_dir'],
            num_clients=config['federated']['num_clients'],
            alpha=config['data']['non_iid_alpha'],
            batch_size=config['federated']['batch_size'],
            num_workers=num_workers
        )

        # 网络拓扑
        self.num_clients = config['federated']['num_clients']
        topology_type = config['topology']['type']
        if topology_type == 'ring':
            self.neighbors = generate_ring_topology(self.num_clients)
        else:
            from utils.topology import generate_mesh_topology
            self.neighbors = generate_mesh_topology(
                self.num_clients,
                degree=config['topology']['degree']
            )

        # 节点划分
        num_byzantine = int(self.num_clients * config['federated']['byzantine_ratio'])
        self.honest_nodes = list(range(self.num_clients - num_byzantine))
        self.byzantine_nodes = list(range(self.num_clients - num_byzantine, self.num_clients))

        print(f"Honest nodes: {len(self.honest_nodes)}, Byzantine nodes: {len(self.byzantine_nodes)}")

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
        """
        训练主循环

        参数:
            method: 'sama' | 'balance' | 'scclip' | 'fedavg' | 'krum' | 'multi_krum' | 'trimmed_mean' | 'coord_median'

        返回:
            dict - 训练历史
        """
        print(f"\nStarting training: {method.upper()}")

        # 初始化模型
        models = [SimpleCNN().to(self.device) for _ in range(self.num_clients)]
        optimizers = [torch.optim.SGD(m.parameters(), lr=lr) for m in models]

        # 初始化聚合器
        if method == 'sama':
            aggregator = SAMAAggregator(
                alpha=self.config['sama']['alpha'],
                use_temperature=self.config['sama']['use_temperature'],
                tau_max=self.config['sama']['tau_max'],
                tau_min=self.config['sama']['tau_min'],
                trust_layers=self.config['sama'].get('trust_layers', None),
            )
        elif method == 'balance':
            aggregator = BALANCEAggregator(
                alpha=self.config['balance']['alpha'],
                gamma=self.config['balance']['gamma'],
                kappa=self.config['balance']['kappa']
            )
        elif method == 'scclip':
            aggregator = SCCLIPAggregator(
                alpha=self.config['scclip']['alpha'],
                clip_constant=self.config['scclip']['clip_constant']
            )
        elif method == 'fedavg':
            aggregator = FedAvgAggregator(
                alpha=self.config['fedavg']['alpha']
            )
        elif method == 'krum':
            aggregator = KrumAggregator(
                alpha=self.config['krum']['alpha'],
                byzantine_ratio=self.config['krum']['byzantine_ratio']
            )
        elif method == 'multi_krum':
            aggregator = KrumAggregator(
                alpha=self.config['multi_krum']['alpha'],
                multi_k=self.config['multi_krum']['multi_k'],
                byzantine_ratio=self.config['multi_krum']['byzantine_ratio']
            )
        elif method == 'trimmed_mean':
            aggregator = TrimmedMeanAggregator(
                alpha=self.config['trimmed_mean']['alpha'],
                trim_ratio=self.config['trimmed_mean']['trim_ratio']
            )
        elif method == 'coord_median':
            aggregator = CoordMedianAggregator(
                alpha=self.config['coord_median']['alpha']
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # 记录
        history = {
            'test_loss': [],
            'test_acc': [],
            'consensus_error': [],
        }

        num_rounds = self.config['federated']['num_rounds']
        local_epochs = self.config['federated']['local_epochs']
        lr = self.config['optimizer']['lr']

        # 训练循环
        pbar = tqdm(range(num_rounds), desc=f"{method.upper()} training")
        for t in pbar:
            # 本地训练
            local_vecs = []
            for i in range(self.num_clients):
                model = models[i]

                if i in self.honest_nodes:
                    # 诚实节点正常训练
                    model.train()
                    for epoch in range(local_epochs):
                        for data, target in self.train_loaders[i]:
                            data, target = data.to(self.device), target.to(self.device)

                            optimizer = optimizers[i]
                            optimizer.zero_grad()
                            output = model(data)
                            loss = torch.nn.functional.cross_entropy(output, target)
                            loss.backward()
                            optimizer.step()
                elif isinstance(self.attack, LabelFlippingAttack):
                    # 拜占庭节点: Label Flipping — 用翻转标签训练
                    model.train()
                    for epoch in range(local_epochs):
                        for data, target in self.train_loaders[i]:
                            data, target = data.to(self.device), target.to(self.device)
                            target = self.attack.flip_labels(target)

                            optimizer = optimizers[i]
                            optimizer.zero_grad()
                            output = model(data)
                            loss = torch.nn.functional.cross_entropy(output, target)
                            loss.backward()
                            optimizer.step()

                local_vecs.append(aggregator.model_to_vector(models[i]))

            # 拜占庭攻击（非Label Flipping类型在训练后修改模型参数）
            if self.attack and not isinstance(self.attack, LabelFlippingAttack):
                honest_vecs_atk = [local_vecs[i] for i in self.honest_nodes]
                if isinstance(self.attack, (OmniscientAttack, KrumAttack, TrimAttack)):
                    for byz_id in self.byzantine_nodes:
                        local_vecs[byz_id] = self.attack.attack(honest_vecs_atk)
                else:
                    for byz_id in self.byzantine_nodes:
                        local_vecs[byz_id] = self.attack.attack(local_vecs[byz_id])

            # 去中心化聚合
            updated_vecs = []
            for i in range(self.num_clients):
                own_vec = local_vecs[i]
                neighbor_vecs = [local_vecs[j] for j in self.neighbors[i]]

                if i in self.honest_nodes:
                    aggregated, agg_stats = aggregator.aggregate(
                        own_vec, neighbor_vecs, t=t, T=num_rounds, return_stats=True
                    )
                    avg_trust = agg_stats.get('avg_trust', None)
                    final_vec = aggregator.final_update(own_vec, aggregated, avg_trust=avg_trust)
                else:
                    final_vec = own_vec

                updated_vecs.append(final_vec)

            # 更新模型
            for i, vec in enumerate(updated_vecs):
                aggregator.load_from_vector(models[i], vec)

            # 评估（每log_interval轮）
            if (t + 1) % self.config['logging']['log_interval'] == 0:
                # 计算共识误差
                honest_vecs = [updated_vecs[i] for i in self.honest_nodes]

                honest_vecs = torch.stack(honest_vecs)
                honest_mean = honest_vecs.mean(dim=0)
                D_t = torch.mean(torch.norm(honest_vecs - honest_mean, dim=1).pow(2)).item()

                # 测试
                global_model = SimpleCNN().to(self.device)
                aggregator.load_from_vector(global_model, honest_mean)
                global_model.eval()

                correct = 0
                total = 0
                loss_sum = 0

                with torch.no_grad():
                    for data, target in self.test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        output = global_model(data)
                        loss_sum += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
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


def run_mnist_experiment(config_path=None):
    """Run MNIST comparison experiment"""
    print("=" * 80)
    print("Experiment C1: MNIST Full Training")
    print("=" * 80)

    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / 'configs' / 'mnist.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print(f"Config file: {config_path}")
    print(f"Experiment name: {config['experiment']['name']}")
    print(f"Number of clients: {config['federated']['num_clients']}")
    print(f"Byzantine ratio: {config['federated']['byzantine_ratio']}")
    print(f"Training rounds: {config['federated']['num_rounds']}")

    # 初始化训练器
    trainer = FederatedTrainer(config)

    # 训练多个方法
    methods = ['sama', 'balance', 'scclip', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'coord_median']
    results = {}

    for method in methods:
        print(f"\n{'='*80}")
        results[method] = trainer.train(method=method)

    # 绘图对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    attack_type = config['attack']['type']
    byz_ratio = config['federated']['byzantine_ratio']
    noniid_alpha = config['data']['non_iid_alpha']
    fig.suptitle(f"MNIST | Attack={attack_type} | Byzantine={byz_ratio:.0%} | Dirichlet α={noniid_alpha}",
                 fontsize=13, fontweight='bold')

    log_interval = config['logging']['log_interval']
    colors = {
        'sama': 'C0', 'balance': 'C1', 'scclip': 'C2',
        'fedavg': 'C3', 'krum': 'C4', 'multi_krum': 'C5',
        'trimmed_mean': 'C6', 'coord_median': 'C7'
    }

    # Top-left: Test loss
    for method in methods:
        t_vals = np.arange(log_interval, config['federated']['num_rounds'] + 1, log_interval)
        axes[0, 0].plot(t_vals, results[method]['test_loss'],
                       label=method.upper(), color=colors[method], linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Training Round')
    axes[0, 0].set_ylabel('Test Loss')
    axes[0, 0].set_title('Test Loss Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Top-right: Test accuracy
    for method in methods:
        t_vals = np.arange(log_interval, config['federated']['num_rounds'] + 1, log_interval)
        axes[0, 1].plot(t_vals, results[method]['test_acc'],
                       label=method.upper(), color=colors[method], linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Training Round')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Consensus error
    for method in methods:
        t_vals = np.arange(log_interval, config['federated']['num_rounds'] + 1, log_interval)
        axes[1, 0].plot(t_vals, results[method]['consensus_error'],
                       label=method.upper(), color=colors[method], linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Training Round')
    axes[1, 0].set_ylabel('Consensus Error $D_t$')
    axes[1, 0].set_title('Consensus Error Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Bottom-right: Final performance bar chart
    final_accs = [results[method]['test_acc'][-1] for method in methods]
    x = np.arange(len(methods))
    bars = axes[1, 1].bar(x, final_accs, color=[colors[m] for m in methods], alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels([m.upper() for m in methods])
    axes[1, 1].set_ylabel('Test Accuracy (%)')
    axes[1, 1].set_title('Final Performance Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 标注数值
    for bar, acc in zip(bars, final_accs):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()

    # Save
    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    fname = f"mnist_{attack_type}_byz{int(byz_ratio*100)}_alpha{noniid_alpha}.png"
    plt.savefig(save_dir / fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {save_dir / fname}")

    # Print final results
    print("\n" + "=" * 80)
    print("MNIST Experiment Final Results")
    print("=" * 80)
    for method in methods:
        final_loss = results[method]['test_loss'][-1]
        final_acc = results[method]['test_acc'][-1]
        final_consensus = results[method]['consensus_error'][-1]
        print(f"{method.upper():8s}: Loss={final_loss:.4f}, Acc={final_acc:.2f}%, D={final_consensus:.4f}")

    return results


if __name__ == "__main__":
    results = run_mnist_experiment()
