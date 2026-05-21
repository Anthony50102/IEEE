#!/bin/bash
#SBATCH -J disco_rollout
#SBATCH -o %x.o%j.log
#SBATCH -e %x.e%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A PHY25003
#SBATCH --mail-type=fail,end

# DISCO long-horizon rollout on Vista GH200.
# Saves full pred + ref fields + Γₙ/Γc time-series as HDF5 for offline plotting.
# See ROADMAP.md Phase 6 (p6-figures) and disco/rollout.py.

set -euo pipefail

module reset
module load gcc/15.1.0 cuda/12.5 python3/3.11.8

source "$WORK/envs/disco/bin/activate"
cd "$WORK/repos/IEEE"

ts="$(date +%Y%m%d_%H%M%S)"
out="$SCRATCH/IEEE/output/${ts}_${SLURM_JOB_ID}_disco_rollout"
mkdir -p "$out"
echo "out=$out"

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

# --- required env vars ----------------------------------------------------
# CONFIG    : training YAML used to fit CHECKPOINT
# CHECKPOINT: absolute path to best.pt or latest.pt
# DATA_ROOT : HW2D data root (default: $WORK/.../hw2d)
# ALPHAS    : space-separated list, e.g. "0.1 1.0 5.0"
# ROLLOUT_STEPS, SEED : optional
#
# Submit:
#   sbatch --export=ALL,CONFIG=configs/disco/hw2d_multi.yaml,\
#       CHECKPOINT=$SCRATCH/IEEE/output/<run>/checkpoints/best.pt,\
#       ALPHAS="0.1 1.0 5.0" \
#       scripts/vista/disco_rollout.slurm

: "${CONFIG:?CONFIG (training yaml) required}"
: "${CHECKPOINT:?CHECKPOINT (.pt) required}"
: "${ALPHAS:?ALPHAS (space-separated list) required}"
DATA_ROOT="${DATA_ROOT:-/work2/10407/anthony50102/frontera/data/IEEE/hw2d}"
ROLLOUT_STEPS="${ROLLOUT_STEPS:-2000}"
SEED="${SEED:-0}"

echo "config=$CONFIG"
echo "checkpoint=$CHECKPOINT"
echo "data_root=$DATA_ROOT"
echo "alphas=$ALPHAS"
echo "rollout_steps=$ROLLOUT_STEPS seed=$SEED"

# shellcheck disable=SC2086  # we want word-split on $ALPHAS
srun python -m disco.rollout \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --alphas $ALPHAS \
    --rollout-steps "$ROLLOUT_STEPS" \
    --seed "$SEED" \
    --out "$out"

echo "done; artifacts in $out"
