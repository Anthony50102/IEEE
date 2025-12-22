#!/bin/bash
#SBATCH -J popinf_pod           # Job name
#SBATCH -o popinf_%j.out    # Output file (%j = job ID)
#SBATCH -e popinf_%j.err    # Error file
#SBATCH -p development                # Queue (partition)
#SBATCH -N 16                     # Number of nodes
#SBATCH -n 32                   # Total MPI tasks (56 cores per node on Frontera)
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
ibrun -n 32 python3 opinf/step_1_parallel_preprocess.py \
    --config config/popinf_10train_2test.yaml \
    --run-dir $SCRATCH/IEEE/output/20251218_150416_hw_rom_experiment 

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
