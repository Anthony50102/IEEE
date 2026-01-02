#!/bin/bash
#=============================================================================
# OpInf Pipeline - Local Development Runner
#
# Usage:
#   ./run_local.sh [step] [config] [run_dir]
#
# Examples:
#   ./run_local.sh 1 config/local.yaml
#   ./run_local.sh 2 config/local.yaml /path/to/run_dir
#   ./run_local.sh 3 config/local.yaml /path/to/run_dir
#   ./run_local.sh all config/local.yaml
#
#=============================================================================

set -e

STEP=${1:-1}
CONFIG=${2:-config/local_opinf.yaml}
RUN_DIR=${3:-}
N_PROCS=${N_PROCS:-4}

cd "$(dirname "$0")/.."

print_header() {
    echo ""
    echo "=============================================="
    echo " $1"
    echo "=============================================="
}

run_step_1() {
    print_header "STEP 1: Preprocessing and POD"
    mpirun -n $N_PROCS python3 opinf/step_1_preprocess.py \
        --config "$CONFIG" \
        --save-pod-energy
}

run_step_2() {
    print_header "STEP 2: ROM Training"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: Run directory required for Step 2"
        exit 1
    fi
    mpirun -n $N_PROCS python3 opinf/step_2_train.py \
        --config "$CONFIG" \
        --run-dir "$RUN_DIR"
}

run_step_3() {
    print_header "STEP 3: Evaluation"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: Run directory required for Step 3"
        exit 1
    fi
    python3 opinf/step_3_evaluate.py \
        --config "$CONFIG" \
        --run-dir "$RUN_DIR"
}

print_header "OpInf Pipeline (Local)"
echo "Step: $STEP"
echo "Config: $CONFIG"
echo "Processes: $N_PROCS"
echo "Start: $(date)"

case $STEP in
    1) run_step_1 ;;
    2) run_step_2 ;;
    3) run_step_3 ;;
    all)
        run_step_1
        # Extract run directory from step 1 output
        echo "Note: For 'all' mode, manually pass run_dir to steps 2 & 3"
        ;;
    *) echo "Invalid step: $STEP"; exit 1 ;;
esac

print_header "Complete"
echo "End: $(date)"
