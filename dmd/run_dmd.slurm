#!/bin/bash
#=============================================================================
# DMD Pipeline - Unified SLURM Launcher for TACC Frontera
#
# Usage:
#   sbatch run_dmd.slurm [step] [config] [run_dir]
#
# Examples:
#   sbatch run_dmd.slurm 1 config/example.yaml
#   sbatch run_dmd.slurm 2 config/example.yaml /path/to/run_dir
#   sbatch run_dmd.slurm 3 config/example.yaml /path/to/run_dir
#   sbatch run_dmd.slurm all config/example.yaml
#
# Steps:
#   1 - Preprocessing and POD computation (parallel)
#   2 - DMD fitting (serial)
#   3 - Evaluation and prediction (serial)
#   all - Run all steps sequentially
#
#=============================================================================

#SBATCH -J dmd_pipeline
#SBATCH -o dmd_%j.out
#SBATCH -e dmd_%j.err
#SBATCH -p nvdimm
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=56
#SBATCH -t 24:00:00

set -e

# =============================================================================
# Configuration
# =============================================================================

STEP=${1:-1}
CONFIG=${2:-config/example.yaml}
RUN_DIR=${3:-}

# Load modules
module load intel/19.1.1
module load impi/19.0.9
module load python3/3.9.2
module load phdf5/1.10.4

# Navigate to project directory
cd $WORK/repos/IEEE

# Install dependencies
pip install -q -c frontera_pip_constraints.txt h5netcdf pydmd

# Set environment
export OMP_NUM_THREADS=56

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
    
    ibrun -n $SLURM_NTASKS python3 dmd/step_1_preprocess.py \
        --config "$CONFIG"
    
    # Capture the run directory from the output
    export RUN_DIR=$(grep "Run directory:" dmd_${SLURM_JOB_ID}.out | tail -1 | awk '{print $NF}')
    echo "Created run directory: $RUN_DIR"
}

run_step_2() {
    print_header "STEP 2: DMD Fitting (Serial)"
    echo "Config: $CONFIG"
    echo "Run dir: $RUN_DIR"
    
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: Run directory required for Step 2"
        exit 1
    fi
    
    python3 dmd/step_2_train.py \
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
    
    python3 dmd/step_3_evaluate.py \
        --config "$CONFIG" \
        --run-dir "$RUN_DIR"
}

# =============================================================================
# Main
# =============================================================================

print_header "DMD Pipeline"
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

# Rename log files to include step information
cd $SLURM_SUBMIT_DIR
if [ -f "dmd_${SLURM_JOB_ID}.out" ]; then
    mv "dmd_${SLURM_JOB_ID}.out" "dmd_${SLURM_JOB_ID}_step${STEP}.out"
fi
if [ -f "dmd_${SLURM_JOB_ID}.err" ]; then
    mv "dmd_${SLURM_JOB_ID}.err" "dmd_${SLURM_JOB_ID}_step${STEP}.err"
fi
echo "Logs renamed to dmd_${SLURM_JOB_ID}_step${STEP}.{out,err}"
