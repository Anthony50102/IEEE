#!/bin/bash
#=============================================================================
# OpInf Pipeline - Unified SLURM Launcher for TACC Frontera
#
# Usage:
#   sbatch run_opinf.slurm [step] [config] [run_dir]
#
# Examples:
#   sbatch run_opinf.slurm 1 config/opinf.yaml
#   sbatch run_opinf.slurm 2 config/opinf.yaml /path/to/run_dir
#   sbatch run_opinf.slurm 3 config/opinf.yaml /path/to/run_dir
#   sbatch run_opinf.slurm all config/opinf.yaml
#
# Steps:
#   1 - Preprocessing and POD computation (parallel)
#   2 - ROM training / hyperparameter sweep (parallel)
#   3 - Evaluation and prediction (serial)
#   all - Run all steps sequentially
#
#=============================================================================

#SBATCH -J opinf_pipeline
#SBATCH -o opinf_%j.out
#SBATCH -e opinf_%j.err
#SBATCH -p normal
#SBATCH -N 2 # Increase for step 1
#SBATCH -n 112
#SBATCH --cpus-per-task=1
#SBATCH -t 4:00:00

# Recommended values
# Step 1: N=2, n=2, cpus=56, t=1:00:00
# Step 2: N=10, 560, cpus=1, t=2:00:00

set -e

# =============================================================================
# Configuration
# =============================================================================

STEP=${1:-1}
CONFIG=${2:-config/opinf.yaml}
RUN_DIR=${3:-}

# Load modules
module load intel/19.1.1
module load impi/19.0.9
module load python3/3.9.2
module load phdf5/1.10.4

# Navigate to project directory
cd $WORK/repos/IEEE

# Install dependencies
pip install -q -c frontera_pip_constraints.txt h5netcdf

# Set environment
export OMP_NUM_THREADS=1

# =============================================================================
# Functions
# =============================================================================

print_header() {
    echo ""
    echo "=============================================="
    echo " $1"
    echo "=============================================="
}

run_step_1() {
    print_header "STEP 1: Preprocessing and POD (Parallel)"
    echo "Config: $CONFIG"
    echo "Nodes: $SLURM_NNODES, Tasks: $SLURM_NTASKS"
    
    ibrun -n $SLURM_NTASKS python3 opinf/step_1_preprocess.py \
        --config "$CONFIG" \
        --save-pod-energy
    
    # Capture the run directory from the output
    export RUN_DIR=$(grep "Run directory:" opinf_${SLURM_JOB_ID}.out | tail -1 | awk '{print $NF}')
    echo "Created run directory: $RUN_DIR"
}

run_step_2() {
    print_header "STEP 2: ROM Training (Parallel)"
    echo "Config: $CONFIG"
    echo "Run dir: $RUN_DIR"
    
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: Run directory required for Step 2"
        exit 1
    fi
    
    ibrun -n $SLURM_NTASKS python3 opinf/step_2_train.py \
        --config "$CONFIG" \
        --run-dir "$RUN_DIR"
}

run_step_3() {
    print_header "STEP 3: Evaluation (Serial)"
    echo "Config: $CONFIG"
    echo "Run dir: $RUN_DIR"
    
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: Run directory required for Step 3"
        exit 1
    fi
    
    python3 opinf/step_3_evaluate.py \
        --config "$CONFIG" \
        --run-dir "$RUN_DIR"
}

# =============================================================================
# Main
# =============================================================================

print_header "OpInf Pipeline"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Step: $STEP"
echo "Config: $CONFIG"

case $STEP in
    1)
        run_step_1
        ;;
    2)
        run_step_2
        ;;
    3)
        run_step_3
        ;;
    all)
        run_step_1
        run_step_2
        run_step_3
        ;;
    *)
        echo "Invalid step: $STEP"
        echo "Valid steps: 1, 2, 3, all"
        exit 1
        ;;
esac

print_header "Complete"
echo "End time: $(date)"
