"""
Visualization Tool: Convergence Curve Plotting
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'


def plot_convergence_comparison(results_dict, save_path=None, title="Convergence Comparison"):
    """
    绘制多方法收敛对比图

    参数:
        results_dict: dict - {method_name: history_dict}
        save_path: Path - 保存路径
        title: str - 图表标题
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    methods = list(results_dict.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    # Test loss
    for idx, method in enumerate(methods):
        history = results_dict[method]
        if 'test_loss' in history:
            axes[0, 0].plot(history['test_loss'], label=method.upper(),
                          color=colors[idx], linewidth=2, alpha=0.7)
    axes[0, 0].set_xlabel('Evaluation Steps')
    axes[0, 0].set_ylabel('Test Loss')
    axes[0, 0].set_title('Test Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Test accuracy
    for idx, method in enumerate(methods):
        history = results_dict[method]
        if 'test_acc' in history:
            axes[0, 1].plot(history['test_acc'], label=method.upper(),
                          color=colors[idx], linewidth=2, alpha=0.7)
    axes[0, 1].set_xlabel('Evaluation Steps')
    axes[0, 1].set_ylabel('Test Accuracy (%)')
    axes[0, 1].set_title('Test Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Consensus error (log scale)
    for idx, method in enumerate(methods):
        history = results_dict[method]
        if 'consensus_error' in history:
            axes[1, 0].plot(history['consensus_error'], label=method.upper(),
                          color=colors[idx], linewidth=2, alpha=0.7)
    axes[1, 0].set_xlabel('Evaluation Steps')
    axes[1, 0].set_ylabel('Consensus Error $D_t$')
    axes[1, 0].set_title('Consensus Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # Final performance comparison
    final_accs = []
    method_labels = []
    for method in methods:
        history = results_dict[method]
        if 'test_acc' in history and len(history['test_acc']) > 0:
            final_accs.append(history['test_acc'][-1])
            method_labels.append(method.upper())

    if final_accs:
        x = np.arange(len(method_labels))
        bars = axes[1, 1].bar(x, final_accs, color=colors[:len(final_accs)], alpha=0.7)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(method_labels)
        axes[1, 1].set_ylabel('Test Accuracy (%)')
        axes[1, 1].set_title('Final Performance')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        # Annotate values
        for bar, acc in zip(bars, final_accs):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    return fig


def plot_training_curves(train_losses, train_accs, val_losses, val_accs,
                         save_path=None, title="Training Progress"):
    """
    Plot training curves

    Parameters:
        train_losses: List[float]
        train_accs: List[float]
        val_losses: List[float]
        val_accs: List[float]
        save_path: Path
        title: str
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(train_losses, label='Train', linewidth=2, alpha=0.7)
    axes[0].plot(val_losses, label='Validation', linewidth=2, alpha=0.7)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(train_accs, label='Train', linewidth=2, alpha=0.7)
    axes[1].plot(val_accs, label='Validation', linewidth=2, alpha=0.7)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")

    return fig


__all__ = ['plot_convergence_comparison', 'plot_training_curves']
