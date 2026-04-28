#!/bin/bash
# SAMA-DFL Unified Experiment Launcher

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$PROJ_DIR/configs/mnist.yaml"
LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "======================================"
echo "  SAMA-DFL Experiment Launcher"
echo "======================================"
echo ""
echo -e "${CYAN}Theory Verification:${NC}"
echo "  [1] Lemma 4.1"
echo "  [2] Convergence Rate"
echo "  [3] Kappa Measurement"
echo "  [4] Consensus Diameter"
echo "  [5] Lyapunov"
echo ""
echo -e "${CYAN}Performance (will ask attack type):${NC}"
echo "  [6] MNIST Comparison"
echo "  [7] CIFAR-10 Comparison"
echo "  [8] Byzantine Sweep"
echo "  [9] Non-IID Sweep"
echo "  [10] Ablation Study"
echo ""
echo -e "${CYAN}Batch:${NC}"
echo "  [A] All Theory (1-5)"
echo "  [B] All Performance (6-10)"
echo "  [0] Everything"
echo ""
read -p "Select experiments (comma-separated, e.g. 1,6,10): " SELECTION

# Expand batch selections
SELECTION=$(echo "$SELECTION" | sed 's/A/1,2,3,4,5/g; s/B/6,7,8,9,10/g; s/0/1,2,3,4,5,6,7,8,9,10/g')
IFS=',' read -ra EXPS <<< "$SELECTION"

# Check if any performance experiment is selected
NEED_ATTACK=false
for e in "${EXPS[@]}"; do
    e=$(echo "$e" | tr -d ' ')
    if [[ "$e" -ge 6 && "$e" -le 10 ]]; then
        NEED_ATTACK=true
        break
    fi
done

ATTACK_TYPE=""
if $NEED_ATTACK; then
    echo ""
    echo -e "${YELLOW}Select attack type:${NC}"
    echo "  [1] gaussian"
    echo "  [2] label_flipping"
    echo "  [3] omniscient"
    read -p "Attack: " ATTACK_CHOICE
    case $ATTACK_CHOICE in
        1) ATTACK_TYPE="gaussian" ;;
        2) ATTACK_TYPE="label_flipping" ;;
        3) ATTACK_TYPE="omniscient" ;;
        *) echo "Unknown attack, defaulting to gaussian"; ATTACK_TYPE="gaussian" ;;
    esac
    # Update config
    sed -i "s/type: \".*\"  #.*attack/type: \"${ATTACK_TYPE}\"  # attack/" "$CONFIG"
    # Fallback: direct match without comment
    sed -i "/^attack:/,/^[^ ]/{s/type: .*/type: \"${ATTACK_TYPE}\"/}" "$CONFIG"
    echo -e "Attack set to: ${GREEN}${ATTACK_TYPE}${NC}"
fi

# Map experiment numbers to commands and log files
declare -A EXP_CMD
declare -A EXP_LOG
declare -A EXP_NAME

EXP_CMD[1]="python $PROJ_DIR/run_experiments.py --experiment lemma41"
EXP_CMD[2]="python $PROJ_DIR/run_experiments.py --experiment convergence"
EXP_CMD[3]="python $PROJ_DIR/run_experiments.py --experiment kappa"
EXP_CMD[4]="python $PROJ_DIR/run_experiments.py --experiment consensus"
EXP_CMD[5]="python $PROJ_DIR/run_experiments.py --experiment lyapunov"
EXP_CMD[6]="python $PROJ_DIR/run_experiments.py --experiment mnist"
EXP_CMD[7]="python $PROJ_DIR/run_experiments.py --experiment cifar10"
EXP_CMD[8]="python $PROJ_DIR/run_experiments.py --experiment byz_sweep"
EXP_CMD[9]="python $PROJ_DIR/run_experiments.py --experiment noniid_sweep"
EXP_CMD[10]="python $PROJ_DIR/run_experiments.py --experiment ablation"

EXP_LOG[1]="lemma41"
EXP_LOG[2]="convergence"
EXP_LOG[3]="kappa"
EXP_LOG[4]="consensus"
EXP_LOG[5]="lyapunov"
EXP_LOG[6]="mnist"
EXP_LOG[7]="cifar10"
EXP_LOG[8]="byz_sweep"
EXP_LOG[9]="noniid_sweep"
EXP_LOG[10]="ablation"

EXP_NAME[1]="Lemma 4.1"
EXP_NAME[2]="Convergence Rate"
EXP_NAME[3]="Kappa"
EXP_NAME[4]="Consensus Diameter"
EXP_NAME[5]="Lyapunov"
EXP_NAME[6]="MNIST"
EXP_NAME[7]="CIFAR-10"
EXP_NAME[8]="Byzantine Sweep"
EXP_NAME[9]="Non-IID Sweep"
EXP_NAME[10]="Ablation"

# Launch experiments in parallel
echo ""
echo "======================================"
echo "  Launching experiments..."
echo "======================================"
echo ""

PIDS=()
LOGS=()
NAMES=()

for e in "${EXPS[@]}"; do
    e=$(echo "$e" | tr -d ' ')
    if [[ -z "${EXP_CMD[$e]}" ]]; then
        echo "Unknown experiment: $e, skipping"
        continue
    fi
    LOGFILE="$LOG_DIR/${EXP_LOG[$e]}.log"
    echo -e "  ${GREEN}[Starting]${NC} ${EXP_NAME[$e]} → logs/${EXP_LOG[$e]}.log"
    ${EXP_CMD[$e]} > "$LOGFILE" 2>&1 &
    PIDS+=($!)
    LOGS+=("$LOGFILE")
    NAMES+=("${EXP_NAME[$e]}")
done

echo ""
echo "======================================"
echo -e "  ${GREEN}${#PIDS[@]} experiments running${NC}"
echo "======================================"
echo ""
echo -e "${CYAN}Monitor progress:${NC}"
for i in "${!NAMES[@]}"; do
    echo "  tail -f logs/${EXP_LOG[${EXPS[$i]// /}]}.log    # ${NAMES[$i]}"
done
echo ""
echo -e "${CYAN}Stop all:${NC}"
echo "  pkill -f run_experiments.py"
echo ""

# Wait for all
echo "Waiting for all experiments to finish..."
echo ""

for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "  ${GREEN}[Done]${NC} ${NAMES[$i]}"
    else
        echo -e "  ${YELLOW}[Failed]${NC} ${NAMES[$i]} (exit code: $EXIT_CODE, check ${LOGS[$i]})"
    fi
done

echo ""
echo "======================================"
echo "  All experiments complete!"
echo "  Results: ls -lh results/"
echo "======================================"
