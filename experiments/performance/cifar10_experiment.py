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
        self.honest_nodes = set(range(self.num_clients - num_byzantine))
        self.byzantine_nodes = set(range(self.num_clients - num_byzantine, self.num_clients))

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
        optimizers = [torch.optim.SGD(m.parameters(), lr=lr,
                                      momentum=self.config['optimizer'].get('momentum', 0.0),
                                      weight_decay=self.config['optimizer'].get('weight_decay', 0.0))
                      for m in models]
        eval_model = SimpleCNN(num_classes=10, in_channels=3).to(self.device)

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
            local_vecs = [None] * self.num_clients
            for i in range(self.num_clients):
                model = models[i]

                if i in self.honest_nodes:
                    model.train()
                    optimizer = optimizers[i]
                    for epoch in range(local_epochs):
                        for data, target in self.train_loaders[i]:
                            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                            optimizer.zero_grad()
                            output = model(data)
                            loss = torch.nn.functional.cross_entropy(output, target)
                            loss.backward()
                            optimizer.step()
                elif self.attack and isinstance(self.attack, LabelFlippingAttack):
                    model.train()
                    optimizer = optimizers[i]
                    for epoch in range(local_epochs):
                        for data, target in self.train_loaders[i]:
                            data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                            target = self.attack.flip_labels(target)
                            optimizer.zero_grad()
                            output = model(data)
                            loss = torch.nn.functional.cross_entropy(output, target)
                            loss.backward()
                            optimizer.step()

                local_vecs[i] = aggregator.model_to_vector(models[i])

            # 拜占庭攻击（非Label Flipping类型）
            if self.attack and not isinstance(self.attack, LabelFlippingAttack):
                honest_vecs_atk = [local_vecs[i] for i in self.honest_nodes]
                if isinstance(self.attack, (OmniscientAttack, KrumAttack, TrimAttack)):
                    for byz_id in self.byzantine_nodes:
                        local_vecs[byz_id] = self.attack.attack(honest_vecs_atk)
                else:
                    for byz_id in self.byzantine_nodes:
                        local_vecs[byz_id] = self.attack.attack(local_vecs[byz_id])

            # 聚合
            all_vecs = torch.stack(local_vecs)
            updated_vecs = [None] * self.num_clients
            for i in range(self.num_clients):
                own_vec = local_vecs[i]
                neighbor_vecs = all_vecs[self.neighbors[i]]

                if i in self.honest_nodes:
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

            # 评估
            if (t + 1) % self.config['logging']['log_interval'] == 0:
                # 共识误差
                honest_vecs = [updated_vecs[i] for i in self.honest_nodes]
                honest_vecs_t = torch.stack(honest_vecs)
                honest_mean = honest_vecs_t.mean(dim=0)
                D_t = torch.mean(torch.norm(honest_vecs_t - honest_mean, dim=1).pow(2)).item()

                # 测试
                global_model = eval_model
                aggregator.load_from_vector(global_model, honest_mean)
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
    plt.close()
    print(f"\nPlot saved to: {save_dir / fname}")

    # 原始数据保存
    import json
    data = {
        'meta': {
            'dataset': 'cifar10',
            'attack_type': attack_type,
            'byzantine_ratio': byz_ratio,
            'noniid_alpha': noniid_alpha,
            'num_rounds': config['federated']['num_rounds'],
            'log_interval': config['logging']['log_interval'],
        },
        'results': {m: {k: (v.tolist() if hasattr(v, 'tolist') else v)
                        for k, v in results[m].items()}
                   for m in results},
    }
    json_path = save_dir / f"cifar10_{attack_type}_byz{int(byz_ratio*100)}_alpha{noniid_alpha}.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Raw data saved to: {json_path.name}")

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
