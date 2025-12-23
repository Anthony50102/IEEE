#!/bin/bash
#SBATCH -J popinf_pod           # Job name
#SBATCH -o popinf_%j.out        # Output file (%j = job ID)
#SBATCH -e popinf_%j.err        # Error file
#SBATCH -p development          # Queue (partition)
#SBATCH -N 16                   # Number of nodes
#SBATCH -n 32                   # Total MPI tasks
#SBATCH --cpus-per-task=28      # Cores per task (56 cores/node ÷ 2 ranks/node)
#SBATCH -t 02:00:00             # Time limit (HH:MM:SS)

# =============================================================================
# OpInf Parallel Hyperparameter Sweep - TACC Frontera
# =============================================================================

# Load required modules
module load intel/19.1.1 
module load impi/19.0.9
module load python3/3.9.2
module load phdf5/1.10.4

# Set OpenMP threads to match cpus-per-task
export OMP_NUM_THREADS=28

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
ibrun -n 32 python3 opinf/step_1_parallel_preprocess.py \
        --config config/popinf_8train_2test.yaml \
        --save-pod-energy


echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
