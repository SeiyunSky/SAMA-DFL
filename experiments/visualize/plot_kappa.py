"""
Visualization Tool: Kappa Value Analysis Plots
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_kappa_comparison(kappa_sama, kappa_balance, save_path=None):
    """
    绘制κ值对比图

    参数:
        kappa_sama: tuple - (mean, std)
        kappa_balance: tuple - (mean, std)
        save_path: Path - 保存路径
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    methods = ['SAMA-DFL', 'BALANCE']
    kappa_means = [kappa_sama[0], kappa_balance[0]]
    kappa_stds = [kappa_sama[1], kappa_balance[1]]

    x = np.arange(len(methods))
    bars = ax.bar(x, kappa_means, yerr=kappa_stds, capsize=5,
                  alpha=0.7, color=['C0', 'C1'])

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Robustness Constant κ')
    ax.set_title('Robustness Constant Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate improvement
    improvement = (1 - kappa_sama[0] / kappa_balance[0]) * 100
    ax.text(0.5, max(kappa_means) * 0.8,
           f'{improvement:.1f}% improvement',
           ha='center', fontsize=12, color='green', fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kappa comparison plot saved: {save_path}")

    return fig


def plot_kappa_evolution(kappa_sama_history, kappa_balance_history, save_path=None):
    """
    Plot kappa value evolution over time

    Parameters:
        kappa_sama_history: List[float]
        kappa_balance_history: List[float]
        save_path: Path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series
    axes[0].plot(kappa_sama_history, label='SAMA-DFL', linewidth=2, alpha=0.7)
    axes[0].plot(kappa_balance_history, label='BALANCE', linewidth=2, alpha=0.7)
    axes[0].axhline(np.mean(kappa_sama_history), color='C0', linestyle='--', alpha=0.5)
    axes[0].axhline(np.mean(kappa_balance_history), color='C1', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Training Round')
    axes[0].set_ylabel('Kappa Value')
    axes[0].set_title('Kappa Evolution Over Training')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribution
    axes[1].hist(kappa_sama_history, bins=30, alpha=0.5, label='SAMA-DFL', color='C0')
    axes[1].hist(kappa_balance_history, bins=30, alpha=0.5, label='BALANCE', color='C1')
    axes[1].axvline(np.mean(kappa_sama_history), color='C0', linestyle='--', linewidth=2)
    axes[1].axvline(np.mean(kappa_balance_history), color='C1', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Kappa Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Kappa Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kappa evolution plot saved: {save_path}")

    return fig


def plot_kappa_heatmap(kappa_matrix, client_labels=None, save_path=None):
    """
    Plot kappa value heatmap (different client pairs)

    Parameters:
        kappa_matrix: np.ndarray - shape (n_clients, n_clients)
        client_labels: List[str] - client labels
        save_path: Path
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(kappa_matrix, cmap='RdYlGn_r', aspect='auto')

    # Labels
    if client_labels is None:
        client_labels = [f"C{i}" for i in range(kappa_matrix.shape[0])]

    ax.set_xticks(np.arange(len(client_labels)))
    ax.set_yticks(np.arange(len(client_labels)))
    ax.set_xticklabels(client_labels)
    ax.set_yticklabels(client_labels)

    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate values
    for i in range(kappa_matrix.shape[0]):
        for j in range(kappa_matrix.shape[1]):
            text = ax.text(j, i, f'{kappa_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontsize=8)

    ax.set_title("Kappa Value Heatmap (Client Pairs)")
    fig.colorbar(im, ax=ax, label='Kappa Value')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Kappa heatmap saved: {save_path}")

    return fig


__all__ = ['plot_kappa_comparison', 'plot_kappa_evolution', 'plot_kappa_heatmap']
