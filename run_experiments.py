"""
Main experiment runner
"""
import argparse
import sys
import importlib
import io
import os
from pathlib import Path

# 强制 stdout/stderr 为行缓冲，TUI 子进程才能实时看到 print/tqdm 输出
# 只在支持 reconfigure 时执行（部分环境下 stdout 是裸二进制流）
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except (AttributeError, io.UnsupportedOperation):
    pass

sys.path.append(str(Path(__file__).parent))


def run_all_theory_experiments():
    """Run all theory verification experiments"""
    print("\n" + "="*80)
    print("Theory Verification Experiments")
    print("="*80 + "\n")

    theory_path = str(Path(__file__).parent / 'experiments' / 'theory_verification')
    sys.path.insert(0, theory_path)

    print("\n[1/5] Lemma 4.1: Magnitude alignment formula verification")
    module = importlib.import_module('1_lemma41_verify')
    module.verify_lemma_41(num_tests=1000, dim=1000)

    print("\n[2/5] B1: Convergence rate measurement")
    module = importlib.import_module('2_convergence_rate')
    module.run_convergence_comparison()

    print("\n[3/5] B2: Kappa measurement")
    module = importlib.import_module('3_kappa_measurement')
    module.run_kappa_measurement()

    print("\n[4/5] B3: Consensus diameter")
    module = importlib.import_module('4_consensus_diameter')
    module.measure_consensus_diameter()

    print("\n[5/5] B4: Lyapunov function verification")
    module = importlib.import_module('5_lyapunov_verify')
    module.run_lyapunov_verification()

    print("\n" + "="*80)
    print("Theory verification complete")
    print("="*80)


def run_all_performance_experiments():
    """Run all performance experiments"""
    print("\n" + "="*80)
    print("Performance Comparison Experiments")
    print("="*80 + "\n")

    perf_path = str(Path(__file__).parent / 'experiments' / 'performance')
    sys.path.insert(0, perf_path)

    print("\n[1/5] C1: MNIST multi-attack table (8 methods × 6 attacks)")
    module = importlib.import_module('multi_attack_table')
    module.run_multi_attack_table()

    print("\n[2/5] C2: CIFAR-10 multi-attack table (8 methods × 6 attacks)")
    module = importlib.import_module('multi_attack_table')
    module.run_cifar10_attack_table()

    print("\n[3/5] C3: Byzantine ratio sweep")
    module = importlib.import_module('sweep_experiments')
    module.run_byzantine_sweep()

    print("\n[4/5] C4: Non-IID level sweep")
    module = importlib.import_module('sweep_experiments')
    module.run_noniid_sweep()

    print("\n[5/5] Ablation study (4 variants)")
    module = importlib.import_module('ablation_study')
    module.run_ablation_study()

    print("\n" + "="*80)
    print("Performance experiments complete")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='SAMA-DFL Experiments')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'theory', 'performance'],
                       help='Run mode: all, theory, performance')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Single experiment: lemma41, convergence, kappa, consensus, lyapunov, '
                            'cifar10_attack_table, multi_attack_table, byz_sweep, noniid_sweep, '
                            'ablation, client_scale')

    args = parser.parse_args()

    if args.experiment:
        theory_path = str(Path(__file__).parent / 'experiments' / 'theory_verification')
        perf_path = str(Path(__file__).parent / 'experiments' / 'performance')

        if args.experiment == 'lemma41':
            sys.path.insert(0, theory_path)
            module = importlib.import_module('1_lemma41_verify')
            module.verify_lemma_41()
        elif args.experiment == 'convergence':
            sys.path.insert(0, theory_path)
            module = importlib.import_module('2_convergence_rate')
            module.run_convergence_comparison()
        elif args.experiment == 'kappa':
            sys.path.insert(0, theory_path)
            module = importlib.import_module('3_kappa_measurement')
            module.run_kappa_measurement()
        elif args.experiment == 'consensus':
            sys.path.insert(0, theory_path)
            module = importlib.import_module('4_consensus_diameter')
            module.measure_consensus_diameter()
        elif args.experiment == 'lyapunov':
            sys.path.insert(0, theory_path)
            module = importlib.import_module('5_lyapunov_verify')
            module.run_lyapunov_verification()
        elif args.experiment == 'mnist':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('mnist_experiment')
            module.run_mnist_experiment()
        elif args.experiment == 'cifar10':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('cifar10_experiment')
            module.run_cifar10_experiment()
        elif args.experiment == 'byz_sweep':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('sweep_experiments')
            module.run_byzantine_sweep()
        elif args.experiment == 'noniid_sweep':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('sweep_experiments')
            module.run_noniid_sweep()
        elif args.experiment == 'ablation':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('ablation_study')
            module.run_ablation_study()
        elif args.experiment == 'cifar10_attack_table':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('multi_attack_table')
            module.run_cifar10_attack_table()
        elif args.experiment == 'multi_attack_table':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('multi_attack_table')
            module.run_multi_attack_table()
        elif args.experiment == 'client_scale':
            sys.path.insert(0, perf_path)
            module = importlib.import_module('client_scale_experiment')
            module.run_client_scale_experiment()
        else:
            print(f"Unknown experiment: {args.experiment}")
            sys.exit(1)
        return

    if args.mode == 'all':
        run_all_theory_experiments()
        run_all_performance_experiments()
    elif args.mode == 'theory':
        run_all_theory_experiments()
    elif args.mode == 'performance':
        run_all_performance_experiments()

    print("\n" + "="*80)
    print("All experiments complete. Results saved to ./results/")
    print("="*80)


if __name__ == "__main__":
    main()
