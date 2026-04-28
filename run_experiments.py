"""
Main experiment runner
"""
import argparse
import sys
import importlib
from pathlib import Path

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

    print("\n[1/4] C1: MNIST (SAMA vs BALANCE vs SCCLIP)")
    module = importlib.import_module('mnist_experiment')
    module.run_mnist_experiment()

    print("\n[2/4] C2: CIFAR-10 (SAMA vs BALANCE vs SCCLIP)")
    module = importlib.import_module('cifar10_experiment')
    module.run_cifar10_experiment()

    print("\n[3/4] C3: Byzantine ratio sweep")
    module = importlib.import_module('sweep_experiments')
    module.run_byzantine_sweep()

    print("\n[4/4] C4: Non-IID level sweep")
    module = importlib.import_module('sweep_experiments')
    module.run_noniid_sweep()

    print("\n" + "="*80)
    print("Performance experiments complete")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='SAMA-DFL Experiments')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'theory', 'performance'],
                       help='Run mode: all, theory, performance')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Single experiment: lemma41, convergence, kappa, consensus, lyapunov, mnist, cifar10, byz_sweep, noniid_sweep, ablation')

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
