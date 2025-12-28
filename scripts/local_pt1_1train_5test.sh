#!/bin/bash
set -e

# Navigate to project directory
cd $HOME/Repositories/IEEE

# Print job info
echo "=============================================="
echo "OpInf Parallel Hyperparameter Sweep"
echo "=============================================="
# echo "Job ID: $SLURM_JOB_ID"
# echo "Nodes: $SLURM_NNODES"
# echo "Tasks: $SLURM_NTASKS"
# echo "CPUs per task: $SLURM_CPUS_PER_TASK"
# echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Start time: $(date)"
echo "=============================================="

# Run parallel sweep
mpirun -n 4 python3 opinf/step_1_parallel_preprocess.py \
        --config config/local_opinf_1train_5test.yaml \
        --save-pod-energy


echo "=============================================="
echo "End time: $(date)"
echo "=============================================="


