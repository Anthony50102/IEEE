#!/bin/bash
#=============================================================================
# OpInf Pipeline - Smart SLURM Submission Script
#
# Automatically sets the right resources for each step.
#
# Usage:
#   ./submit_opinf.sh <step> <config> [run_dir]
#
# Examples:
#   ./submit_opinf.sh 1 config/opinf.yaml
#   ./submit_opinf.sh 2 config/opinf.yaml /path/to/run_dir
#   ./submit_opinf.sh 3 config/opinf.yaml /path/to/run_dir
#
#=============================================================================

set -e

STEP=${1:?Usage: ./submit_opinf.sh <step> <config> [run_dir]}
CONFIG=${2:?Usage: ./submit_opinf.sh <step> <config> [run_dir]}
RUN_DIR=${3:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set resources based on step
case $STEP in
    1)
        # Step 1: Preprocessing - few nodes, many threads per task
        NODES=2
        NTASKS=2
        CPUS=56
        TIME="1:00:00"
        OMP_THREADS=56
        ;;
    2)
        # Step 2: Hyperparameter sweep - many tasks, 1 thread each
        NODES=10
        NTASKS=560
        CPUS=1
        TIME="2:00:00"
        OMP_THREADS=1
        ;;
    3)
        # Step 3: Evaluation - single node, serial
        NODES=1
        NTASKS=1
        CPUS=56
        TIME="2:00:00"
        OMP_THREADS=56
        ;;
    *)
        echo "Invalid step: $STEP"
        echo "Valid steps: 1, 2, 3"
        exit 1
        ;;
esac

echo "=============================================="
echo " Submitting OpInf Step $STEP"
echo "=============================================="
echo "Config: $CONFIG"
echo "Run dir: ${RUN_DIR:-<will be created>}"
echo "Resources: $NODES nodes, $NTASKS tasks, $CPUS cpus/task, $TIME"
echo ""

# Build sbatch command with resource overrides
SBATCH_CMD="sbatch"
SBATCH_CMD="$SBATCH_CMD --nodes=$NODES"
SBATCH_CMD="$SBATCH_CMD --ntasks=$NTASKS"
SBATCH_CMD="$SBATCH_CMD --cpus-per-task=$CPUS"
SBATCH_CMD="$SBATCH_CMD --time=$TIME"
SBATCH_CMD="$SBATCH_CMD --export=ALL,OMP_NUM_THREADS=$OMP_THREADS"
SBATCH_CMD="$SBATCH_CMD --job-name=opinf_step${STEP}"

# Submit
if [ -z "$RUN_DIR" ]; then
    $SBATCH_CMD "$SCRIPT_DIR/run_opinf.slurm" "$STEP" "$CONFIG"
else
    $SBATCH_CMD "$SCRIPT_DIR/run_opinf.slurm" "$STEP" "$CONFIG" "$RUN_DIR"
fi
