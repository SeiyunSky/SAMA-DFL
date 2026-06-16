"""
Experiment 1: Lemma 4.1 Numerical Verification
Verify magnitude alignment distance formula: ||w_tilde_j - w_i||^2 = 2||w_i||^2(1 - cos(w_i, w_j))
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set matplotlib to avoid font issues
plt.rcParams['font.family'] = 'DejaVu Sans'


def verify_lemma_41(num_tests=1000, dim=1000):
    """
    数值验证引理4.1

    参数:
        num_tests: 测试向量对数量
        dim: 向量维度

    返回:
        dict - 验证结果统计
    """
    print("=" * 60)
    print("Experiment A: Lemma 4.1 Numerical Verification")
    print("=" * 60)

    errors = []

    for test_id in range(num_tests):
        # 生成随机向量
        w_i = torch.randn(dim)
        w_j = torch.randn(dim)

        # 计算幅度
        norm_i = torch.norm(w_i)
        norm_j = torch.norm(w_j)

        if norm_i < 1e-8 or norm_j < 1e-8:
            continue

        # 幅度对齐
        w_tilde_j = norm_i * (w_j / norm_j)

        # 左侧: 直接计算距离的平方
        left_side = torch.norm(w_tilde_j - w_i).pow(2).item()

        # 右侧: 理论公式
        cos_sim = torch.dot(w_i, w_j) / (norm_i * norm_j)
        right_side = 2 * norm_i.pow(2).item() * (1 - cos_sim.item())

        # 计算误差
        abs_error = abs(left_side - right_side)
        rel_error = abs_error / (right_side + 1e-10)

        errors.append(rel_error)

        # Print first 5 samples
        if test_id < 5:
            print(f"\nSample {test_id + 1}:")
            print(f"  ||w_i|| = {norm_i.item():.4f}")
            print(f"  ||w_j|| = {norm_j.item():.4f}")
            print(f"  cos(w_i, w_j) = {cos_sim.item():.4f}")
            print(f"  Left (actual) = {left_side:.6f}")
            print(f"  Right (theory) = {right_side:.6f}")
            print(f"  Relative error = {rel_error:.8f}")

    errors = np.array(errors)

    # 统计结果
    results = {
        'mean_error': np.mean(errors),
        'max_error': np.max(errors),
        'std_error': np.std(errors),
        'percentile_95': np.percentile(errors, 95),
        'percentile_99': np.percentile(errors, 99)
    }

    print("\n" + "=" * 60)
    print("Verification Results (Relative Error)")
    print("=" * 60)
    print(f"Test samples: {len(errors)}")
    print(f"Mean error:   {results['mean_error']:.2e}")
    print(f"Max error:    {results['max_error']:.2e}")
    print(f"Std dev:      {results['std_error']:.2e}")
    print(f"95th percentile: {results['percentile_95']:.2e}")
    print(f"99th percentile: {results['percentile_99']:.2e}")

    # Success criteria
    success = results['mean_error'] < 1e-6 and results['percentile_99'] < 1e-4
    print(f"\nVerification: {'PASS' if success else 'FAIL'}")

    # Plot error distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(np.log10(errors + 1e-12), bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('log10(Relative Error)')
    plt.ylabel('Frequency')
    plt.title('Lemma 4.1 Numerical Verification - Error Distribution')
    plt.axvline(np.log10(results['mean_error']), color='r', linestyle='--',
                label=f"Mean: {results['mean_error']:.2e}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save
    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    plt.savefig(save_dir / 'lemma41_verification.png', dpi=300, bbox_inches='tight')
    plt.close()
    import json
    save_dir = Path(__file__).parent.parent.parent / 'results'
    save_dir.mkdir(exist_ok=True)
    json_path = save_dir / 'lemma41_verification.json'
    with open(json_path, 'w') as f:
        json.dump({
            'results': {k: float(v) for k, v in results.items()},
            'errors': errors.tolist(),
        }, f, indent=2)
    print(f"Raw data saved: {json_path.name}")
    print(f"\nPlot saved to: {save_dir / 'lemma41_verification.png'}")

    return results


if __name__ == "__main__":
    results = verify_lemma_41(num_tests=1000, dim=1000)
