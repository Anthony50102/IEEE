#!/bin/bash
# =============================================================================
# Manifold Regularization Sweep Launcher
# =============================================================================
#
# Submits individual SLURM jobs for each regularization value.
# First job (reg index 0) also computes POD baseline.
#
# Usage:
#   ./mani_parallel_sweep.sh
#
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION - EDIT THESE
# =============================================================================
DATA_FILE="/scratch2/10407/anthony50102/IEEE/data/hw2d_sim/t600_d512x512_striped/test_nu5e-9.h5"
OUTPUT_BASE="/scratch2/10407/anthony50102/IEEE/output/mani_sweep"
RUN_NAME="mani_reg_sweep"

# Regularization values (one job per value)
REG_VALUES=(1e3 1e4 1e5 1e6 1e7 1e8 1e9 1e10)

# Manifold parameters
R=84
N_CHECK=200

# Snapshot ranges
TRAIN_START=8000
TRAIN_END=16000
TEST_START=16000
TEST_END=20000
TRAINING_END=8000

# Physics
DT=0.025
N_FIELDS=2
NX=512
NY=512

# SLURM settings
PARTITION="normal"
TIME="20:00:00"
NODES=1
CPUS=56

# =============================================================================
# SUBMIT JOBS
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=============================================="
echo "Manifold Regularization Sweep"
echo "=============================================="
echo "Project: $PROJECT_DIR"
echo "Output: $OUTPUT_BASE"
echo "Run name: $RUN_NAME"
echo "Reg values: ${REG_VALUES[*]}"
echo "=============================================="

for i in "${!REG_VALUES[@]}"; do
    REG="${REG_VALUES[$i]}"
    JOB_NAME="mani_reg${i}"
    
    # First job computes POD baseline
    if [ "$i" -eq 0 ]; then
        WITH_POD="--with-pod"
        echo "Submitting job $i (reg=$REG) WITH POD baseline..."
    else
        WITH_POD=""
        echo "Submitting job $i (reg=$REG)..."
    fi
    
    sbatch <<EOF
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -o ${OUTPUT_BASE}/${JOB_NAME}_%j.out
#SBATCH -e ${OUTPUT_BASE}/${JOB_NAME}_%j.err
#SBATCH -p ${PARTITION}
#SBATCH -N ${NODES}
#SBATCH -n 1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH -t ${TIME}

set -e

module load intel/19.1.1 
module load impi/19.0.9
module load python3/3.9.2
module load phdf5/1.10.4

export OMP_NUM_THREADS=${CPUS}

cd ${PROJECT_DIR}

pip install -c frontera_pip_constraints.txt h5netcdf xarray scipy --quiet

echo "=============================================="
echo "Manifold Experiment - reg=${REG}"
echo "=============================================="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "=============================================="

python3 analysis/mani_parallel_experiment.py \\
    --data-file "${DATA_FILE}" \\
    --output-base "${OUTPUT_BASE}" \\
    --run-name "${RUN_NAME}" \\
    --reg ${REG} \\
    --r ${R} \\
    --n-check ${N_CHECK} \\
    --train-start ${TRAIN_START} \\
    --train-end ${TRAIN_END} \\
    --test-start ${TEST_START} \\
    --test-end ${TEST_END} \\
    --training-end ${TRAINING_END} \\
    --dt ${DT} \\
    --n-fields ${N_FIELDS} \\
    --nx ${NX} \\
    --ny ${NY} \\
    ${WITH_POD}

echo "=============================================="
echo "End time: \$(date)"
echo "=============================================="
EOF

done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"
echo "Collect results after completion with:"
echo "  python analysis/collect_mani_results.py --output-base $OUTPUT_BASE --run-name $RUN_NAME"
