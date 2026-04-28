# SAMA-DFL: Self-Anchored Magnitude-Aligned Decentralized Federated Learning

**Byzantine-Robust Decentralized Federated Learning - Experiment Codebase**

## Project Structure

```
SAMA_DFL_Experiments/
├── aggregators/          # Aggregation algorithms
│   ├── base.py          # Abstract base class
│   ├── sama.py          # SAMA-DFL (proposed)
│   ├── balance.py       # BALANCE baseline
│   └── scclip.py        # SCCLIP baseline
├── attacks/             # Byzantine attack implementations
│   └── __init__.py      # Gaussian, LabelFlipping, Omniscient
├── models/              # Neural network models
│   └── __init__.py      # SimpleCNN, MLP
├── utils/               # Utilities
│   ├── data_loader.py   # Non-IID Dirichlet partition
│   ├── topology.py      # Network topology + spectral gap
│   ├── metrics.py       # Evaluation metrics
│   └── logger.py        # Logging system
├── experiments/
│   ├── theory_verification/   # 5 theory experiments
│   │   ├── 1_lemma41_verify.py
│   │   ├── 2_convergence_rate.py
│   │   ├── 3_kappa_measurement.py
│   │   ├── 4_consensus_diameter.py
│   │   └── 5_lyapunov_verify.py
│   └── performance/           # Performance experiments
│       ├── mnist_experiment.py
│       ├── cifar10_experiment.py
│       └── sweep_experiments.py    # Byzantine ratio & Non-IID sweeps
├── configs/
│   ├── mnist.yaml       # Main config (MNIST + theory verification)
│   └── cifar10.yaml     # CIFAR-10 overrides
├── run_experiments.py   # CLI experiment runner
├── run_parallel.sh      # Parallel execution script
├── setup.sh             # Environment setup
├── monitor.py           # GPU resource monitor
└── requirements.txt
```

---

## Quick Start

```bash
# Environment setup
bash setup.sh

# Run all experiments
python run_experiments.py --mode all

# Run specific experiment groups
python run_experiments.py --mode theory        # Theory verification (~30 min)
python run_experiments.py --mode performance   # Performance comparison (~2 hours)

# Run individual experiments
python run_experiments.py --experiment mnist
python run_experiments.py --experiment cifar10
python run_experiments.py --experiment kappa
python run_experiments.py --experiment byz_sweep
python run_experiments.py --experiment noniid_sweep

# Parallel execution (recommended for GPU server)
bash run_parallel.sh
```

---

## Experiments

### Theory Verification

| Experiment | Target | Success Criterion |
|------------|--------|-------------------|
| Lemma 4.1 | Distance formula accuracy | Relative error < 0.1% |
| Convergence rate | lambda = mu*eta fitting | Error < 20% |
| Kappa comparison | kappa_SAMA < kappa_BALANCE | Reduction > 15% |
| Consensus diameter | R_inf theoretical bound | Error < 30% |
| Lyapunov | Phi_t monotone decrease | Geometric decay + steady state |

### Performance Comparison

| Dataset | Methods | Attacks | Non-IID | Byzantine Ratios |
|---------|---------|---------|---------|-----------------|
| MNIST | SAMA / BALANCE / SCCLIP | Gaussian, LabelFlipping | Dirichlet alpha=0.1 | 20% |
| CIFAR-10 | SAMA / BALANCE / SCCLIP | Gaussian | Dirichlet alpha=0.1 | 20% |

### Parameter Sweeps

| Sweep | Fixed | Variable | Methods |
|-------|-------|----------|---------|
| C3: Byzantine ratio | MNIST, alpha=0.1 | f/n in {0.1, 0.2, 0.3, 0.4} | SAMA / BALANCE / SCCLIP |
| C4: Non-IID level | MNIST, f/n=0.2 | alpha in {0.1, 0.3, 0.5, 1.0} | SAMA / BALANCE / SCCLIP |

---

## Key Parameters

- **Clients**: n=20, Byzantine ratio=20%
- **Self-anchor weight**: alpha=0.5 (theoretically optimal)
- **Non-IID**: Dirichlet alpha=0.1 (high heterogeneity)
- **Topology**: Ring (sparse) / Mesh (dense)
- **Rounds**: 250 (MNIST), 400 (CIFAR-10)

---

## Results

All outputs saved to `results/`:

| File | Content |
|------|---------|
| `lemma41_verification.png` | Lemma 4.1 error distribution |
| `kappa_measurement.png` | Kappa time series + comparison |
| `convergence_rate.png` | Convergence rate fitting |
| `consensus_diameter.png` | Consensus diameter vs theoretical bound |
| `lyapunov_verification.png` | Lyapunov function decay |
| `mnist_experiment.png` | MNIST loss/accuracy/consensus comparison |
| `cifar10_experiment.png` | CIFAR-10 comparison |
| `byzantine_sweep.png` | Accuracy vs Byzantine ratio |
| `noniid_sweep.png` | Accuracy vs Dirichlet alpha |
