#!/bin/bash
#SBATCH -J opinf_sweep           # Job name
#SBATCH -o opinf_sweep_%j.out    # Output file (%j = job ID)
#SBATCH -e opinf_sweep_%j.err    # Error file
#SBATCH -p development                # Queue (partition)
#SBATCH -N 10                     # Number of nodes
#SBATCH -n 560                   # Total MPI tasks (56 cores per node on Frontera)
#SBATCH -t 02:00:00              # Time limit (HH:MM:SS)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=anthony50102@tacc.utexas.edu

# =============================================================================
# OpInf Parallel Hyperparameter Sweep - TACC Frontera
# =============================================================================
# 
# Usage:
#   sbatch submit_sweep.sh
#
# Notes:
#   - Frontera CLX nodes have 56 cores each
#   - Adjust -N and -n based on problem size
#   - Larger grids benefit from more nodes
#
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
cd $WORK/repos/SciML_ROMs_Hasegawa_Wakatani

# Install package in development mode (if not already done)
pip install -e . --no-deps
pip install -c constraints.txt h5netcdf

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
ibrun python3 scripts_anthony/parallel_sweep.py \
    --config cluster \
    --method threshold \
    --threshold-mean 0.05 \
    --threshold-std 0.30

# Alternative: top-k selection
# ibrun python -m scripts_anthony.parallel_sweep \
#     --config cluster \
#     --method top_k \
#     --num-top-models 20

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
