#!/bin/bash
# SAMA-DFL Experiment Runner
# Usage: bash run.sh [experiment] [attack]
#   experiment: lemma41 | convergence | kappa | consensus | lyapunov | mnist | cifar10 | byz_sweep | noniid_sweep | ablation | all
#   attack:     gaussian | label_flipping | omniscient | node_failure (for performance experiments)

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"
mkdir -p logs results

EXP="${1:-}"
ATTACK="${2:-}"

# Interactive mode if no arguments
if [ -z "$EXP" ]; then
    echo ""
    echo "SAMA-DFL Experiment Runner"
    echo "=========================="
    echo ""
    echo "Theory Verification:"
    echo "  lemma41      Lemma 4.1 magnitude alignment formula"
    echo "  convergence  Convergence rate fitting (B1)"
    echo "  kappa        Kappa comparison SAMA vs BALANCE (B2)"
    echo "  consensus    Consensus diameter vs theoretical bound (B3)"
    echo "  lyapunov     Lyapunov monotone decay (B4)"
    echo ""
    echo "Performance:"
    echo "  mnist        MNIST full comparison (all methods x all attacks)"
    echo "  cifar10      CIFAR-10 generalization"
    echo "  byz_sweep    Byzantine ratio sweep C3"
    echo "  noniid_sweep Non-IID level sweep C4"
    echo "  ablation     Ablation study"
    echo ""
    echo "Batch:"
    echo "  theory       Run lemma41+convergence+kappa+consensus+lyapunov"
    echo "  performance  Run mnist+cifar10+byz_sweep+noniid_sweep+ablation"
    echo "  all          Run everything"
    echo ""
    read -p "Experiment: " EXP
fi

# Attack selection for performance experiments
PERF_EXPS="mnist cifar10 byz_sweep noniid_sweep ablation performance all"
if [[ " $PERF_EXPS " == *" $EXP "* ]] && [ -z "$ATTACK" ]; then
    echo ""
    echo "Attack type:"
    echo "  1) gaussian"
    echo "  2) label_flipping"
    echo "  3) omniscient"
    echo "  4) node_failure"
    echo "  (leave blank to use config default)"
    read -p "Attack [1-4 or Enter]: " ATTACK_CHOICE
    case $ATTACK_CHOICE in
        1) ATTACK="gaussian" ;;
        2) ATTACK="label_flipping" ;;
        3) ATTACK="omniscient" ;;
        4) ATTACK="node_failure" ;;
        *) ATTACK="" ;;
    esac
fi

# Pass attack as env var (experiments read ATTACK_TYPE env if set)
if [ -n "$ATTACK" ]; then
    export ATTACK_TYPE="$ATTACK"
    echo "Attack: $ATTACK"
fi

# Expand batch shortcuts
run_exp() {
    local exp="$1"
    local logfile="logs/${exp}.log"
    echo "[$(date '+%H:%M:%S')] Starting: $exp"
    python run_experiments.py --experiment "$exp" > "$logfile" 2>&1
    local code=$?
    if [ $code -eq 0 ]; then
        echo "[$(date '+%H:%M:%S')] Done:    $exp"
    else
        echo "[$(date '+%H:%M:%S')] FAILED:  $exp (see $logfile)"
    fi
}

case "$EXP" in
    theory)
        for e in lemma41 convergence kappa consensus lyapunov; do
            run_exp "$e" &
        done
        wait
        ;;
    performance)
        for e in mnist cifar10 byz_sweep noniid_sweep ablation; do
            run_exp "$e"
        done
        ;;
    all)
        for e in lemma41 convergence kappa consensus lyapunov; do
            run_exp "$e" &
        done
        wait
        for e in mnist cifar10 byz_sweep noniid_sweep ablation; do
            run_exp "$e"
        done
        ;;
    lemma41|convergence|kappa|consensus|lyapunov|mnist|cifar10|byz_sweep|noniid_sweep|ablation)
        run_exp "$EXP"
        ;;
    *)
        echo "Unknown experiment: $EXP"
        echo "Run without arguments for interactive mode."
        exit 1
        ;;
esac

echo ""
echo "Results: ls results/"
