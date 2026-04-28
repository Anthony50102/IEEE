#!/bin/bash
#SBATCH -J setup-data
#SBATCH -o setup_data.o%j
#SBATCH -e setup_data.e%j
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# Copy all HW2D data from $WORK archive to $SCRATCH with Lustre striping.
# Submit with: sbatch scripts/setup_data.slurm

cd $WORK/repos/IEEE
bash scripts/setup_data.sh
