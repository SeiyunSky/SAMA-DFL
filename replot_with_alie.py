"""
replot_with_alie.py
-------------------
从已有的 data_mnist_byz20_alpha0.1.json 加载原始 7 攻击数据，
再从 ALIE log 文件中解析新数据，合并后重新生成全套图表。

用法：
    python replot_with_alie.py \
        --json   "路径/data_mnist_byz20_alpha0.1.json" \
        --logdir "ALIE log 所在目录" \
        --outdir "输出图表目录"

ALIE log 文件命名规则（与现有 log 一致）：
    mnist_{method}_alie.log
例：mnist_sama_alie.log, mnist_balance_alie.log, ...

log 格式（每行一条）：
    round=10/150 acc=85.23%
"""
import json
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'

METHOD_KEYS   = ['sama', 'balance', 'scclip', 'fedavg', 'krum', 'multi_krum', 'trimmed_mean', 'coord_median']
METHOD_LABELS = ['SAMA-DFL', 'BALANCE', 'SC-CLIP', 'FedAvg', 'Krum', 'Multi-Krum', 'Trim-Mean', 'Coord-Med']
METHOD_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                 '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
METHOD_LINESTYLES = ['-', '-', '-', '--', '--', '--', '-.', '-.']

# 攻击顺序：原 7 种 + alie 插在 omniscient 后面（位置 4）
ATTACK_KEYS_OLD   = ['none', 'gaussian', 'label_flipping', 'omniscient',
                     'krum_attack', 'trim_attack', 'broken_node']
ATTACK_KEYS_NEW   = ['none', 'gaussian', 'label_flipping', 'omniscient', 'alie',
                     'krum_attack', 'trim_attack', 'broken_node']
ATTACK_LABELS_NEW = ['No Attack', 'Gaussian', 'Label Flip', 'Omniscient', 'ALIE',
                     'Krum Atk', 'Trim Atk', 'Broken Node']
ATTACK_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#e377c2',
                 '#9467bd', '#8c564b', '#17becf']
ATTACK_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


def parse_log(log_path: Path):
    """从 log 文件解析 accuracy history，返回 List[float]"""
    history = []
    pattern = re.compile(r'acc=([\d.]+)%')
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                history.append(float(m.group(1)))
    return history


def load_alie_logs(logdir: Path):
    """
    从 logdir 读取所有 mnist_{method}_alie.log，
    返回 {method_key: history_list}
    """
    alie_data = {}
    for method in METHOD_KEYS:
        log_file = logdir / f"mnist_{method}_alie.log"
        if log_file.exists():
            history = parse_log(log_file)
            alie_data[method] = history
            print(f"  Loaded ALIE log: {log_file.name}  (rounds={len(history)}, final={history[-1]:.2f}%)")
        else:
            print(f"  [WARN] Missing: {log_file.name} — will fill with zeros")
            alie_data[method] = []
    return alie_data


def merge_data(old_json: Path, alie_logs: dict):
    """
    合并旧 JSON 数据 + ALIE 数据，返回 (final_acc_matrix, histories_matrix)
    维度：[n_methods, n_attacks_new]，攻击顺序按 ATTACK_KEYS_NEW
    """
    with open(old_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data['meta']
    round_ticks = meta['round_ticks']
    n_methods = len(METHOD_KEYS)
    n_attacks = len(ATTACK_KEYS_NEW)

    final_acc = np.zeros((n_methods, n_attacks))
    histories  = [[None] * n_attacks for _ in range(n_methods)]

    for m_idx, method in enumerate(METHOD_KEYS):
        for a_idx, attack in enumerate(ATTACK_KEYS_NEW):
            if attack == 'alie':
                hist = alie_logs.get(method, [])
                acc  = hist[-1] if hist else 0.0
            else:
                hist = data['histories'].get(method, {}).get(attack, [])
                acc  = data['final_acc'].get(method, {}).get(attack, 0.0)
            final_acc[m_idx, a_idx] = acc
            histories[m_idx][a_idx] = hist

    return meta, round_ticks, final_acc, histories


def plot_all(meta, round_ticks, final_acc, histories, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    byz    = meta['byzantine_ratio']
    alpha  = meta['noniid_alpha']
    ds     = meta['dataset'].upper()
    tag    = f"{meta['dataset']}_byz{int(byz*100)}_alpha{alpha}"

    n_methods = len(METHOD_KEYS)
    n_attacks = len(ATTACK_KEYS_NEW)

    # ── heatmap ───────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(final_acc, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
    plt.colorbar(im, ax=ax, label='Test Accuracy (%)')
    ax.set_xticks(range(n_attacks))
    ax.set_xticklabels(ATTACK_LABELS_NEW, fontsize=10)
    ax.set_yticks(range(n_methods))
    ax.set_yticklabels(METHOD_LABELS, fontsize=10)
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Aggregation Method', fontsize=12)
    ax.set_title(
        f'{ds}: Accuracy (%) by Method × Attack\n'
        f'Byzantine={byz:.0%}, Dirichlet α={alpha}',
        fontsize=12, fontweight='bold'
    )
    for m_i in range(n_methods):
        for a_i in range(n_attacks):
            val = final_acc[m_i, a_i]
            color = 'black' if 30 < val < 80 else 'white'
            ax.text(a_i, m_i, f'{val:.1f}', ha='center', va='center',
                    fontsize=8, color=color, fontweight='bold')
    plt.tight_layout()
    p = outdir / f'heatmap_{tag}.png'
    plt.savefig(p, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {p.name}")

    # ── per-attack 曲线（8张）─────────────────────────────────────
    for a_idx, (attack_key, attack_label) in enumerate(zip(ATTACK_KEYS_NEW, ATTACK_LABELS_NEW)):
        fig, ax = plt.subplots(figsize=(8, 5))
        for m_idx in range(n_methods):
            hist = histories[m_idx][a_idx]
            if hist:
                ticks = round_ticks[:len(hist)]
                ax.plot(ticks, hist,
                        label=METHOD_LABELS[m_idx],
                        color=METHOD_COLORS[m_idx],
                        linestyle=METHOD_LINESTYLES[m_idx],
                        linewidth=1.8, marker='o', markersize=3)
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(
            f'{ds} — {attack_label}\nByzantine={byz:.0%}, Dirichlet α={alpha}',
            fontsize=12, fontweight='bold'
        )
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9, ncol=2, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = outdir / f'acc_per_attack_{attack_key}_{tag}.png'
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {p.name}")

    # ── per-method 曲线（8张）────────────────────────────────────
    for m_idx in range(n_methods):
        fig, ax = plt.subplots(figsize=(8, 5))
        for a_idx in range(n_attacks):
            hist = histories[m_idx][a_idx]
            if hist:
                ticks = round_ticks[:len(hist)]
                ax.plot(ticks, hist,
                        label=ATTACK_LABELS_NEW[a_idx],
                        color=ATTACK_COLORS[a_idx],
                        linestyle=ATTACK_LINESTYLES[a_idx],
                        linewidth=1.8, marker='o', markersize=3)
        ax.set_xlabel('Communication Round', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(
            f'{ds} — {METHOD_LABELS[m_idx]}\nByzantine={byz:.0%}, Dirichlet α={alpha}',
            fontsize=12, fontweight='bold'
        )
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9, ncol=2, loc='lower right')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = outdir / f'acc_per_method_{METHOD_KEYS[m_idx]}_{tag}.png'
        plt.savefig(p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {p.name}")

    # ── 控制台汇总表 ──────────────────────────────────────────────
    col_w = 10
    print('\n' + '=' * (12 + col_w * n_attacks))
    print(f"{'Method':>12}", end='')
    for lbl in ATTACK_LABELS_NEW:
        print(f"  {lbl:>{col_w-2}}", end='')
    print()
    print('-' * (12 + col_w * n_attacks))
    for m_idx, lbl in enumerate(METHOD_LABELS):
        print(f"{lbl:>12}", end='')
        for a_idx in range(n_attacks):
            print(f"  {final_acc[m_idx, a_idx]:>{col_w-2}.2f}%", end='')
        print()

    # ── 更新后的 JSON 保存 ────────────────────────────────────────
    merged = {
        'meta': {**meta, 'attacks': ATTACK_KEYS_NEW},
        'final_acc': {
            METHOD_KEYS[m]: {ATTACK_KEYS_NEW[a]: float(final_acc[m, a])
                             for a in range(n_attacks)}
            for m in range(n_methods)
        },
        'histories': {
            METHOD_KEYS[m]: {ATTACK_KEYS_NEW[a]: histories[m][a]
                             for a in range(n_attacks)}
            for m in range(n_methods)
        },
    }
    p = outdir / f'data_{tag}_with_alie.json'
    with open(p, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged JSON saved: {p.name}")


def main():
    parser = argparse.ArgumentParser(description='Replot all figures with ALIE attack added')
    parser.add_argument('--json',   required=True,
                        help='旧的 data_mnist_byz20_alpha0.1.json 路径')
    parser.add_argument('--logdir', required=True,
                        help='存放 mnist_{method}_alie.log 的目录')
    parser.add_argument('--outdir', required=True,
                        help='输出图表目录')
    args = parser.parse_args()

    json_path = Path(args.json)
    logdir    = Path(args.logdir)
    outdir    = Path(args.outdir)

    print(f"JSON : {json_path}")
    print(f"Logs : {logdir}")
    print(f"Out  : {outdir}\n")

    print("Loading ALIE logs...")
    alie_data = load_alie_logs(logdir)

    print("\nMerging data...")
    meta, round_ticks, final_acc, histories = merge_data(json_path, alie_data)

    print("\nGenerating plots...")
    plot_all(meta, round_ticks, final_acc, histories, outdir)

    print("\nDone.")


if __name__ == '__main__':
    main()
