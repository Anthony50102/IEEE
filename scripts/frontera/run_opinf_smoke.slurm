#!/bin/bash
#SBATCH -J opinf_smoke
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 56
#SBATCH -t 00:45:00
#SBATCH -o logs/opinf_smoke_%j.out
#SBATCH -e logs/opinf_smoke_%j.err
#
# OpInf B1 smoke (steps 1+2 only) on a single HW DNS trajectory.
#
# Submit:
#   cd $WORK/repos/IEEE
#   mkdir -p logs
#   sbatch scripts/frontera/run_opinf_smoke.slurm
#
# To run a different config, override CONFIG when submitting:
#   sbatch --export=ALL,CONFIG=configs/opinf/b1_alpha0.1.yaml \
#          scripts/frontera/run_opinf_smoke.slurm
#
# Step 3 (evaluation, serial) is now portable -- reads gamma_n / gamma_c
# directly via h5py.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source scripts/frontera/env.sh

CONFIG="${CONFIG:-configs/opinf/b1_alpha1.yaml}"

# Match Numba/OMP to physical cores per task. With 56 MPI ranks on one
# CLX node, OMP=1 is correct: each rank owns one core.
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1

echo "==> host:    $(hostname)"
echo "==> python:  $(which python3)  $(python3 --version)"
echo "==> CONFIG:  $CONFIG"
echo "==> ranks:   $SLURM_NTASKS"
echo "==> start:   $(date -Iseconds)"

# Ensure opinf imports resolve (utils.py is sibling-imported, not packaged).
export PYTHONPATH="$PWD/opinf:${PYTHONPATH:-}"

# ------------------------------------------------------------------
# Step 1 -- preprocess + POD
# ------------------------------------------------------------------
echo ""
echo "================== STEP 1 =================="
ibrun -n "$SLURM_NTASKS" python3 opinf/step_1_preprocess.py \
    --config "$CONFIG" \
    --save-pod-energy

# step_1 prints "Run directory: <path>" on rank 0; capture for step 2.
RUN_DIR=$(grep -m1 "Run directory:" "logs/opinf_smoke_${SLURM_JOB_ID}.out" \
          | awk -F'Run directory:' '{print $2}' | xargs)
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: could not parse Run directory from step 1 output."
    exit 3
fi
echo "==> RUN_DIR: $RUN_DIR"

# ------------------------------------------------------------------
# Step 2 -- regularizer sweep / model selection
# ------------------------------------------------------------------
echo ""
echo "================== STEP 2 =================="
ibrun -n "$SLURM_NTASKS" python3 opinf/step_2_train.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR"

# ------------------------------------------------------------------
# Step 3 -- evaluation (serial; runs on rank 0 of the allocation)
# ------------------------------------------------------------------
echo ""
echo "================== STEP 3 =================="
python3 opinf/step_3_evaluate.py \
    --config "$CONFIG" \
    --run-dir "$RUN_DIR"

echo ""
echo "==> end:     $(date -Iseconds)"
echo "==> RUN_DIR (final): $RUN_DIR"
