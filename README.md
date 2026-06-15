# SAMA-DFL: Self-Anchored Magnitude-Aligned Decentralized Federated Learning

**Byzantine-Robust Decentralized Federated Learning - Experiment Codebase**

## Project Structure

```
SAMA_DFL_Experiments/
├── aggregators/          # Aggregation algorithms
│   ├── base.py          # Abstract base class (shared final_update)
│   ├── sama.py          # SAMA-DFL (proposed)
│   ├── balance.py       # BALANCE baseline
│   ├── scclip.py        # SC-CLIP baseline
│   ├── fedavg.py        # FedAvg baseline
│   ├── krum.py          # Krum / Multi-Krum baseline
│   ├── trimmed_mean.py  # Trimmed Mean baseline
│   └── coord_median.py  # Coordinate Median baseline
├── attacks/             # Byzantine attack implementations
│   └── __init__.py      # NoAttack, Gaussian, LabelFlipping, Omniscient,
│                        # KrumAttack, TrimAttack
├── models/              # Neural network models
│   └── __init__.py      # SimpleCNN (1-ch MNIST / 3-ch CIFAR-10), MLP
├── utils/               # Utilities
│   ├── data_loader.py   # Non-IID Dirichlet partition;
│   │                    # check_dataset / check_all_datasets / download_dataset
│   ├── topology.py      # Network topology + spectral gap
│   ├── metrics.py       # Evaluation metrics
│   └── logger.py        # Logging system
├── experiments/
│   ├── theory_verification/   # Theory verification (Lemma 4.1 → Theorem 5.2)
│   │   ├── 1_lemma41_verify.py
│   │   ├── 2_convergence_rate.py
│   │   ├── 3_kappa_measurement.py
│   │   ├── 4_consensus_diameter.py
│   │   └── 5_lyapunov_verify.py
│   └── performance/           # Performance experiments
│       ├── multi_attack_table.py   # 8-method × 6-attack table (MNIST + CIFAR-10)
│       ├── sweep_experiments.py    # Byzantine ratio & Non-IID sweeps (8 methods)
│       ├── client_scale_experiment.py  # n ∈ {20,30,40} scalability (8 methods)
│       └── ablation_study.py       # 4 variants incl. HardThreshold
├── configs/
│   ├── mnist.yaml       # Main config (MNIST + theory verification)
│   └── cifar10.yaml     # CIFAR-10 config
├── run_experiments.py   # CLI experiment dispatcher
├── tui.py               # Interactive TUI dashboard (Rich-based, parallel execution)
├── run.sh               # Unified launcher (interactive or batch)
├── setup.sh             # Environment setup (AutoDL/GPU server)
├── monitor.py           # GPU resource monitor
└── requirements.txt
```

---

## Quick Start

```bash
# Environment setup
bash setup.sh

# Interactive TUI dashboard (recommended, supports parallel execution)
python tui.py

# Direct CLI invocation
python run_experiments.py --experiment multi_attack_table   # MNIST 8×6
python run_experiments.py --experiment cifar10_attack_table # CIFAR-10 8×6
python run_experiments.py --experiment byz_sweep
python run_experiments.py --experiment noniid_sweep
python run_experiments.py --experiment client_scale
python run_experiments.py --experiment ablation

# Theory experiments
python run_experiments.py --experiment lemma41
python run_experiments.py --experiment convergence
python run_experiments.py --experiment kappa
python run_experiments.py --experiment consensus
python run_experiments.py --experiment lyapunov

# Batch modes
python run_experiments.py --mode theory        # All 5 theory experiments
python run_experiments.py --mode performance   # All 5 performance experiments
python run_experiments.py --mode all           # Everything

# Override attack type via environment variable (sweep / client_scale)
ATTACK_TYPE=omniscient python run_experiments.py --experiment byz_sweep
ATTACK_TYPE=krum_attack python run_experiments.py --experiment client_scale
```

---

## TUI Dashboard (`tui.py`)

Launch with `python tui.py`. Features:

- **Dataset status panel** — detects missing datasets at startup and offers one-click download with live progress
- **GPU panel** — real-time memory / utilization / temperature
- **Experiment menu** — grouped by A (theory), B (performance), C (sweep)
- **Parallel execution** — up to 4 concurrent jobs (configured for 11 GB VRAM)
- **Live output streaming** — colored log lines during experiment execution
- **Results table** — lists recently generated `.png` files with size and timestamp

TUI menu layout:

| Group | ID | Experiment |
|-------|----|------------|
| A | A1–A5 | Theory verification experiments |
| B | B1 | MNIST multi-attack table (8 methods × 6 attacks) |
| B | B2 | CIFAR-10 multi-attack table (8 methods × 6 attacks) |
| B | B3 | Ablation study (4 variants) |
| B | B4 | Client scalability (n = 20/30/40) |
| C | C1 | Byzantine ratio sweep (0.1–0.4) |
| C | C2 | Non-IID sweep (α = 0.1/0.2/0.3) |

---

## Experiments

### Theory Verification

| Experiment | Target | Success Criterion |
|------------|--------|-------------------|
| Lemma 4.1 (`lemma41`) | Distance formula accuracy | Relative error < 1e-6 |
| Convergence rate (`convergence`) | λ ≈ μη fitting | Error < 20% |
| Kappa comparison (`kappa`) | κ_SAMA < κ_BALANCE | Reduction > 20% |
| Consensus diameter (`consensus`) | R_inf < theoretical bound | Measured < bound |
| Lyapunov (`lyapunov`) | Φ_t monotone decrease | >70% rounds ΔΦ<0, steady-state RSD<5% |

### Performance Comparison (8 Methods × 6 Attacks)

Both MNIST and CIFAR-10 run all 8 aggregation methods against all 6 attack types.

**Methods:** SAMA, BALANCE, SC-CLIP, FedAvg, Krum, Multi-Krum, Trim-Mean, CoordMed

**Attacks:** No Attack, Gaussian, Label Flip, Omniscient, Krum Attack, Trim Attack

**Output per dataset (15 PNG files):**

| File pattern | Content |
|---|---|
| `heatmap_{ds}_byz{N}_alpha{α}.png` | Summary heatmap: method × attack accuracy matrix |
| `acc_per_attack_{atk}_{ds}_....png` | 6 files — per attack, 8 method convergence curves |
| `acc_per_method_{method}_{ds}_....png` | 8 files — per method, 6 attack convergence curves |

### Parameter Sweeps

| Sweep | Fixed | Variable | Methods |
|-------|-------|----------|---------|
| Byzantine ratio (`byz_sweep`) | MNIST, α=0.1, config attack | f/n ∈ {0.1, 0.2, 0.3, 0.4} | All 8 |
| Non-IID level (`noniid_sweep`) | MNIST, f/n=0.2, config attack | α ∈ {0.1, 0.2, 0.3} | All 8 |
| Client scale (`client_scale`) | MNIST, f/n=0.2, α=0.1 | n ∈ {20, 30, 40} | All 8 |

Each sweep generates one PNG with 8 method curves.

### Ablation Study (`ablation`)

| Variant | Disabled component |
|---------|--------------------|
| Full SAMA | — (baseline) |
| No direction trust | φ_j = 1 (uniform weights) |
| No magnitude alignment | Skip Step 2 |
| Hard threshold | Binary cos>0 filter instead of soft weighting |
| No self-anchor | α = 0 |

---

## Attack Types

| Key | Class | Type | Description |
|-----|-------|------|-------------|
| `none` | `NoAttack` | — | No attack; Byzantine nodes train normally |
| `gaussian` | `GaussianAttack` | Black-box | Add Gaussian noise (σ=10) to model parameters |
| `label_flipping` | `LabelFlippingAttack` | Black-box | Train on flipped labels (y → 9-y) |
| `omniscient` | `OmniscientAttack` | White-box | Send amplified negation of honest mean |
| `krum_attack` | `KrumAttack` | White-box | Binary search to minimize Krum score (Fang et al. 2020) |
| `trim_attack` | `TrimAttack` | White-box | Per-dimension push beyond trimmed boundary (Fang et al. 2020) |

White-box attacks receive the full list of honest model updates.
For sweep/scale experiments, override attack at runtime via `ATTACK_TYPE=<key>`.

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Clients | n=20 (default) | Sweep: n ∈ {20,30,40} |
| Byzantine ratio | 20% | 4 Byzantine nodes at n=20 |
| Non-IID | Dirichlet α=0.1 | High heterogeneity; sweep: α ∈ {0.1,0.2,0.3} |
| Topology | Mesh degree=6 | Shared across all methods within one experiment |
| Rounds | 150 (sweep/table), 400 (CIFAR-10 full) | |
| Seed | 42 | Fixed globally for reproducibility |
| SAMA α | 0.5 | Self-anchor weight |
| SAMA trust_layers | [fc2.weight, fc2.bias] | Classification head cosine trust |
| BALANCE γ | 3.0 | |
| SCCLIP clip_constant | 0.1 | |
| Krum byzantine_ratio | 0.2 | Matches experiment Byzantine ratio |
| TrimmedMean trim_ratio | 0.1 | Fraction trimmed from each end |

---

## Reproducibility

- **Seed** (default 42) is set via `configs/mnist.yaml → experiment.seed`
- All methods within the same experiment share **one fixed topology** generated before the sweep loop
- Each sweep level re-seeds + re-generates topology independently, so different byz_ratio / alpha levels are still comparable within each level

---

## Results

All outputs saved to `results/`:

| File | Content |
|------|---------|
| `lemma41_verification.png` | Lemma 4.1 error distribution |
| `kappa_measurement.png` | Kappa comparison SAMA vs BALANCE |
| `convergence_rate.png` | Convergence rate fitting |
| `consensus_diameter.png` | Consensus diameter vs theoretical bound |
| `lyapunov_verification.png` | Lyapunov function decay |
| `heatmap_{ds}_byz{N}_alpha{α}.png` | Method × attack accuracy heatmap |
| `acc_per_attack_{atk}_{ds}_....png` | Per-attack convergence curves (6 per dataset) |
| `acc_per_method_{method}_{ds}_....png` | Per-method convergence curves (8 per dataset) |
| `byzantine_sweep_{attack}_alpha{α}.png` | Accuracy vs Byzantine ratio (8 methods) |
| `noniid_sweep_{attack}_byz{N}.png` | Accuracy vs Dirichlet α (8 methods) |
| `client_scale_{attack}_byz{N}_alpha{α}.png` | Accuracy vs #clients (8 methods) |
| `ablation_study_{attack}.png` | Component ablation results |
