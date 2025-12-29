#!/bin/bash
#SBATCH -J mani_step_1           # Job name
#SBATCH -o mani_step1_%j.out        # Output file (%j = job ID)
#SBATCH -e mani_step1_%j.err        # Error file
#SBATCH -p normal          # Queue (partition)
#SBATCH -N 1                   # Number of nodes
#SBATCH -t 10:00:00             # Time limit (HH:MM:SS)

# =============================================================================
# OpInf Parallel Hyperparameter Sweep - TACC Frontera
# =============================================================================

set -e

# Load required modules
module load intel/19.1.1 
module load impi/19.0.9
module load python3/3.9.2
module load phdf5/1.10.4

# Set OpenMP threads to match cpus-per-task
export OMP_NUM_THREADS=56

# Navigate to project directory
cd $WORK/repos/IEEE

# Install package in development mode (if not already done)
pip install -c frontera_pip_constraints.txt h5netcdf

# Print job info
echo "=============================================="
echo "OpInf Parallel Hyperparameter Sweep"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "Start time: $(date)"
echo "=============================================="

# Run parallel sweep
python3 mani_opinf/step_1.py \
        --config config/mani_opinf_1train_5test.yaml \


echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
