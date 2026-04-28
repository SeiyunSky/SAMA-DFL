"""
Visualization Tool: Robustness Analysis Plots
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_robustness_under_attacks(results_by_attack, save_path=None):
    """
    Plot robustness comparison under different attacks

    Parameters:
        results_by_attack: dict - {attack_name: {method: accuracy}}
        save_path: Path
    """
    attacks = list(results_by_attack.keys())
    methods = list(results_by_attack[attacks[0]].keys())

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(attacks))
    width = 0.35 / len(methods)

    for idx, method in enumerate(methods):
        accuracies = [results_by_attack[attack][method] for attack in attacks]
        offset = width * (idx - len(methods)/2 + 0.5)
        bars = ax.bar(x + offset, accuracies, width, label=method.upper(), alpha=0.7)

        # Annotate values
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Robustness Under Different Attacks')
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Robustness comparison plot saved: {save_path}")

    return fig


def plot_byzantine_ratio_sensitivity(ratios, results_by_ratio, save_path=None):
    """
    Plot Byzantine ratio sensitivity analysis

    Parameters:
        ratios: List[float] - Byzantine ratio list
        results_by_ratio: dict - {method: List[accuracy]}
        save_path: Path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_ratio)))

    for idx, (method, accuracies) in enumerate(results_by_ratio.items()):
        ax.plot(ratios, accuracies, marker='o', label=method.upper(),
               color=colors[idx], linewidth=2, markersize=8, alpha=0.7)

    ax.set_xlabel('Byzantine Ratio f')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Sensitivity to Byzantine Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotate key points
    for ratio in [0.2, 0.3, 0.4]:
        if ratio in ratios:
            ax.axvline(ratio, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Byzantine ratio sensitivity plot saved: {save_path}")

    return fig


def plot_heterogeneity_impact(alphas, results_by_alpha, save_path=None):
    """
    Plot data heterogeneity impact analysis

    Parameters:
        alphas: List[float] - Dirichlet parameter list
        results_by_alpha: dict - {method: List[accuracy]}
        save_path: Path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results_by_alpha)))

    for idx, (method, accuracies) in enumerate(results_by_alpha.items()):
        ax.plot(alphas, accuracies, marker='s', label=method.upper(),
               color=colors[idx], linewidth=2, markersize=8, alpha=0.7)

    ax.set_xlabel('Dirichlet Parameter α (smaller = more heterogeneous)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Impact of Data Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # Annotate IID/Non-IID regions
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.3, label='Highly Non-IID')
    ax.axvline(1.0, color='green', linestyle='--', alpha=0.3, label='Nearly IID')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heterogeneity impact plot saved: {save_path}")

    return fig


def plot_aggregation_weights(weights_history, labels=None, save_path=None):
    """
    Plot aggregation weight evolution

    Parameters:
        weights_history: np.ndarray - shape (rounds, num_neighbors)
        labels: List[str] - neighbor labels
        save_path: Path
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    num_neighbors = weights_history.shape[1]
    if labels is None:
        labels = [f"Neighbor {i+1}" for i in range(num_neighbors)]

    for i in range(num_neighbors):
        ax.plot(weights_history[:, i], label=labels[i], linewidth=2, alpha=0.7)

    ax.set_xlabel('Training Round')
    ax.set_ylabel('Aggregation Weight')
    ax.set_title('Evolution of Aggregation Weights')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight evolution plot saved: {save_path}")

    return fig


__all__ = [
    'plot_robustness_under_attacks',
    'plot_byzantine_ratio_sensitivity',
    'plot_heterogeneity_impact',
    'plot_aggregation_weights'
]
