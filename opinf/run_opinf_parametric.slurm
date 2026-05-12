#!/bin/bash
#=============================================================================
# Parametric OpInf (B2) - Unified SLURM Launcher for TACC Frontera
#
# Usage:
#   sbatch run_opinf_parametric.slurm [step] [config] [run_dir]
#
# Examples:
#   sbatch run_opinf_parametric.slurm 1 configs/opinf/b2_alpha_p015.yaml
#   sbatch run_opinf_parametric.slurm 2 configs/opinf/b2_alpha_p015.yaml <RUN_DIR>
#   sbatch run_opinf_parametric.slurm 3 configs/opinf/b2_alpha_p015.yaml <RUN_DIR>
#   sbatch run_opinf_parametric.slurm all configs/opinf/b2_alpha_p015.yaml
#=============================================================================

#SBATCH -J opinf_param
#SBATCH -o opinf_param_%j.out
#SBATCH -e opinf_param_%j.err
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --cpus-per-task=56
#SBATCH -t 1:00:00

: ${OMP_NUM_THREADS:=56}
export OMP_NUM_THREADS

set -e

STEP=${1:-1}
CONFIG=${2:-configs/opinf/b2_alpha_p015.yaml}
RUN_DIR=${3:-}

module load intel/19.1.1
module load impi/19.0.9
module load python3/3.9.2
module load phdf5/1.10.4

cd $WORK/repos/IEEE
pip install -q -c frontera_pip_constraints.txt h5netcdf

print_header() {
    echo ""
    echo "=============================================="
    echo " $1"
    echo "=============================================="
}

run_step_1() {
    print_header "STEP 1 (parametric): Multi-alpha load + pooled POD"
    echo "Config: $CONFIG"
    echo "Nodes: $SLURM_NNODES, Tasks: $SLURM_NTASKS"
    ibrun -n $SLURM_NTASKS python3 opinf/step_1_preprocess_parametric.py \
        --config "$CONFIG"
}

run_step_2() {
    print_header "STEP 2 (parametric): 2D Tikhonov sweep"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: Run directory required for Step 2"
        exit 1
    fi
    ibrun -n $SLURM_NTASKS python3 opinf/step_2_train_parametric.py \
        --config "$CONFIG" --run-dir "$RUN_DIR"
}

run_step_3() {
    print_header "STEP 3 (parametric): G1 + G3 evaluation"
    if [ -z "$RUN_DIR" ]; then
        echo "ERROR: Run directory required for Step 3"
        exit 1
    fi
    python3 opinf/step_3_evaluate_parametric.py \
        --config "$CONFIG" --run-dir "$RUN_DIR"
}

print_header "Parametric OpInf (B2) Pipeline"
echo "Job ID: $SLURM_JOB_ID  Step: $STEP  Config: $CONFIG"

case $STEP in
    1) run_step_1 ;;
    2) run_step_2 ;;
    3) run_step_3 ;;
    all)
        run_step_1
        # Step 1 prints the run dir; parse from log and pass to step 2/3.
        RUN_DIR=$(grep "Run directory:" opinf_param_${SLURM_JOB_ID}.out \
                  | tail -1 | awk '{print $NF}')
        echo "Run dir: $RUN_DIR"
        run_step_2
        run_step_3
        ;;
    *)
        echo "Invalid step: $STEP"
        exit 1
        ;;
esac

print_header "Complete"
echo "End time: $(date)"
