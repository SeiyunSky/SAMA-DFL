#!/bin/bash
# Parallel experiment runner for GPU server

echo "=========================================="
echo "SAMA-DFL Parallel Experiments"
echo "=========================================="

mkdir -p logs results

echo ""
echo "Launching 3 parallel task groups..."
echo ""

# Task 1: Fast theory verification (background)
echo "[Task 1] Lemma 4.1 + Kappa measurement..."
(
    cd experiments/theory_verification
    python 1_lemma41_verify.py > ../../logs/lemma41.log 2>&1
    python 3_kappa_measurement.py > ../../logs/kappa.log 2>&1
    cd ../..
    echo "[Task 1] Done: Lemma 4.1 + Kappa"
) &
PID1=$!

# Task 2: MNIST training (background)
echo "[Task 2] MNIST full training (SAMA vs BALANCE vs SCCLIP)..."
(
    cd experiments/performance
    python mnist_experiment.py > ../../logs/mnist.log 2>&1
    cd ../..
    echo "[Task 2] Done: MNIST"
) &
PID2=$!

# Task 3: Other theory verification (background)
echo "[Task 3] Convergence + Consensus + Lyapunov..."
(
    cd experiments/theory_verification
    python 2_convergence_rate.py > ../../logs/convergence.log 2>&1
    python 4_consensus_diameter.py > ../../logs/consensus.log 2>&1
    python 5_lyapunov_verify.py > ../../logs/lyapunov.log 2>&1
    cd ../..
    echo "[Task 3] Done: Convergence + Consensus + Lyapunov"
) &
PID3=$!

echo ""
echo "3 tasks launched:"
echo "  Task 1 PID: $PID1 (Theory: Lemma 4.1 + Kappa)"
echo "  Task 2 PID: $PID2 (MNIST training)"
echo "  Task 3 PID: $PID3 (Theory: Convergence + Consensus + Lyapunov)"
echo ""
echo "Monitor logs:"
echo "  tail -f logs/lemma41.log"
echo "  tail -f logs/mnist.log"
echo "  tail -f logs/convergence.log"
echo ""
echo "Waiting for all tasks..."

wait $PID1
wait $PID2
wait $PID3

echo ""
echo "=========================================="
echo "Phase 1 complete!"
echo "=========================================="
echo ""
echo "Results: ls -lh results/"
echo ""
echo "Next steps (optional):"
echo "  python run_experiments.py --experiment cifar10"
echo "  python run_experiments.py --experiment byz_sweep"
echo "  python run_experiments.py --experiment noniid_sweep"
echo ""
echo "=========================================="
