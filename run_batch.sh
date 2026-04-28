#!/bin/bash
# Batch experiment runner
# Reads experiments.txt, runs each config, saves results to dated folders

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$PROJ_DIR/configs/mnist.yaml"
RESULTS_BASE="$PROJ_DIR/results"
LOG_DIR="$PROJ_DIR/logs"
BATCH_FILE="$PROJ_DIR/experiments.txt"
DATE=$(date +%Y%m%d)

GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

echo ""
echo "======================================"
echo "  SAMA-DFL Batch Experiment Runner"
echo "======================================"
echo ""

# Check experiments.txt exists
if [ ! -f "$BATCH_FILE" ]; then
    echo -e "${RED}Error: experiments.txt not found${NC}"
    exit 1
fi

# Show experiment list
echo -e "${CYAN}Experiments defined in experiments.txt:${NC}"
echo ""
IDX=0
declare -a ATTACKS STDS AMPS DESCS
while IFS=',' read -r attack std amp desc || [ -n "$attack" ]; do
    # Skip comments and empty lines
    attack=$(echo "$attack" | tr -d ' ')
    [[ "$attack" == \#* ]] && continue
    [[ -z "$attack" ]] && continue

    std=$(echo "$std" | tr -d ' ')
    amp=$(echo "$amp" | tr -d ' ')
    desc=$(echo "$desc" | tr -d ' ')

    ATTACKS[$IDX]="$attack"
    STDS[$IDX]="$std"
    AMPS[$IDX]="$amp"
    DESCS[$IDX]="$desc"

    echo "  [$IDX] $desc (attack=$attack, std=$std, C=$amp)"
    IDX=$((IDX + 1))
done < "$BATCH_FILE"

TOTAL=$IDX
echo ""
echo "  Total: $TOTAL experiments"
echo ""

# Ask which to run
echo -e "${YELLOW}Select experiments to run:${NC}"
echo "  Enter numbers comma-separated (e.g. 0,3,5)"
echo "  Enter 'all' to run everything"
echo "  Enter 'range' like 0-3 for range"
read -p "Selection: " SEL

# Parse selection
declare -a SELECTED
if [ "$SEL" == "all" ]; then
    for ((i=0; i<TOTAL; i++)); do SELECTED+=($i); done
elif [[ "$SEL" == *-* ]]; then
    START=$(echo "$SEL" | cut -d'-' -f1)
    END=$(echo "$SEL" | cut -d'-' -f2)
    for ((i=START; i<=END; i++)); do SELECTED+=($i); done
else
    IFS=',' read -ra SELECTED <<< "$SEL"
fi

# Ask experiment type
echo ""
echo -e "${YELLOW}Select experiment type:${NC}"
echo "  [1] MNIST performance (mnist)"
echo "  [2] Kappa measurement (kappa)"
echo "  [3] Both (mnist + kappa)"
read -p "Type: " EXP_TYPE

declare -a EXP_NAMES
case $EXP_TYPE in
    1) EXP_NAMES=("mnist") ;;
    2) EXP_NAMES=("kappa") ;;
    3) EXP_NAMES=("mnist" "kappa") ;;
    *) echo "Unknown, defaulting to mnist"; EXP_NAMES=("mnist") ;;
esac

# Backup original config
cp "$CONFIG" "$CONFIG.bak"

echo ""
echo "======================================"
echo "  Running ${#SELECTED[@]} x ${#EXP_NAMES[@]} experiments"
echo "======================================"
echo ""

for idx in "${SELECTED[@]}"; do
    idx=$(echo "$idx" | tr -d ' ')

    ATTACK="${ATTACKS[$idx]}"
    STD="${STDS[$idx]}"
    AMP="${AMPS[$idx]}"
    DESC="${DESCS[$idx]}"

    # Create result folder
    RESULT_DIR="$RESULTS_BASE/${DATE}_${DESC}"
    mkdir -p "$RESULT_DIR"

    # Update config
    sed -i "/^attack:/,/^[^ ]/{
        s|type: .*|type: \"${ATTACK}\"|
        s|gaussian_std: .*|gaussian_std: ${STD}|
        s|amplification: .*|amplification: ${AMP}|
    }" "$CONFIG"

    # Snapshot config
    cp "$CONFIG" "$RESULT_DIR/config_snapshot.yaml"

    # Write params.txt
    cat > "$RESULT_DIR/params.txt" << PARAMS
Experiment: $DESC
Date: $(date '+%Y-%m-%d %H:%M:%S')
Attack: $ATTACK
Gaussian STD: $STD
Amplification (C): $AMP
---
SAMA: alpha=0.5, trust_layers=[fc2.weight, fc2.bias], adaptive_alpha=false, use_temperature=false
BALANCE: gamma=$(grep 'gamma:' "$CONFIG" | head -1 | awk '{print $2}'), kappa_decay=$(grep 'kappa:' "$CONFIG" | head -1 | awk '{print $2}')
SCCLIP: clip_constant=$(grep 'clip_constant:' "$CONFIG" | head -1 | awk '{print $2}')
---
Topology: $(grep 'type:' "$CONFIG" | head -1 | awk '{print $2}')
Degree: $(grep 'degree:' "$CONFIG" | head -1 | awk '{print $2}')
Clients: $(grep 'num_clients:' "$CONFIG" | head -1 | awk '{print $2}')
Byzantine ratio: $(grep 'byzantine_ratio:' "$CONFIG" | head -1 | awk '{print $2}')
Non-IID alpha: $(grep 'non_iid_alpha:' "$CONFIG" | head -1 | awk '{print $2}')
Rounds: $(grep 'num_rounds:' "$CONFIG" | head -1 | awk '{print $2}')
LR: $(grep 'lr:' "$CONFIG" | head -1 | awk '{print $2}')
PARAMS

    for EXP in "${EXP_NAMES[@]}"; do
        LOGFILE="$LOG_DIR/${DESC}_${EXP}.log"

        echo -e "${GREEN}[Running]${NC} $DESC / $EXP"
        echo "  Config: attack=$ATTACK, std=$STD, C=$AMP"
        echo "  Output: $RESULT_DIR/"
        echo "  Log:    $LOGFILE"

        python "$PROJ_DIR/run_experiments.py" --experiment "$EXP" > "$LOGFILE" 2>&1
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo -e "  ${GREEN}[Done]${NC} $DESC / $EXP"
        else
            echo -e "  ${RED}[Failed]${NC} $DESC / $EXP (exit code: $EXIT_CODE)"
        fi

        # Move result images to experiment folder
        for img in "$RESULTS_BASE"/*.png; do
            [ -f "$img" ] && mv "$img" "$RESULT_DIR/" 2>/dev/null
        done
    done

    echo ""
done

# Restore original config
mv "$CONFIG.bak" "$CONFIG"

echo "======================================"
echo "  Batch complete!"
echo "  Results in: $RESULTS_BASE/${DATE}_*/"
echo "======================================"
echo ""
echo "View results:"
echo "  ls results/${DATE}_*/"
