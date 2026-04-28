#!/bin/bash
#SBATCH -J opinf_sweep           # Job name
#SBATCH -o opinf_sweep_%j.out    # Output file (%j = job ID)
#SBATCH -e opinf_sweep_%j.err    # Error file
#SBATCH -p development                # Queue (partition)
#SBATCH -N 16                     # Number of nodes
#SBATCH -n 896                   # Total MPI tasks (56 cores per node on Frontera)
#SBATCH --cpus-per-task=1       # CPUs per task (28 threads per core on Frontera) 
#SBATCH -t 02:00:00              # Time limit (HH:MM:SS)

# =============================================================================
# OpInf Parallel Hyperparameter Sweep - TACC Frontera
# =============================================================================

# Load required modules
module load intel/19.1.1 
module load impi/19.0.9
module load python3/3.9.2
module load phdf5/1.10.4

# Activate conda/virtual environment if needed
# source activate your_env
# OR
# source /path/to/venv/bin/activate

# Navigate to project directory
cd $WORK/repos/IEEE

# Install package in development mode (if not already done)
pip install -c frontera_pip_constraints.txt h5netcdf

# Set OpenMP threads to match cpus-per-task
export OMP_NUM_THREADS=1

# Print job info
echo "=============================================="
echo "OpInf Parallel Hyperparameter Sweep"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks: $SLURM_NTASKS"
echo "Start time: $(date)"
echo "=============================================="

# Run parallel sweep
# Use ibrun for TACC systems (wrapper for mpirun)
ibrun -n 896 python3 opinf/step_2_train_rom.py \
    --config config/opinf_8train_2test.yaml \
    --run-dir $SCRATCH/IEEE/output/20251223_212257_8train_2test 

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
