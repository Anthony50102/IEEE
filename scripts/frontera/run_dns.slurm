#!/bin/bash
#SBATCH -J hw_dns
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o logs/hw_dns_%j.out
#SBATCH -e logs/hw_dns_%j.err
#
# One HW DNS trajectory.
#
# Required env vars (set when submitting via sbatch --export):
#   CONFIG    path to a configs/data/hw_alpha*.yaml (relative to repo root)
#   OUTDIR    where to write trajectory.h5 + data_card.yaml
#
# Submit example (one alpha):
#   cd $WORK/IEEE
#   mkdir -p logs
#   sbatch --export=ALL,CONFIG=configs/data/hw_alpha1.0_n512.yaml,\
#OUTDIR=$SCRATCH/IEEE/data/hw2d/alpha1.0_n512 \
#        scripts/frontera/run_dns.slurm

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

# Canonical Frontera environment.
source scripts/frontera/env.sh

if [ -z "${CONFIG:-}" ] || [ -z "${OUTDIR:-}" ]; then
    echo "ERROR: must set CONFIG and OUTDIR env vars (see header)."
    exit 2
fi

echo "==> host:    $(hostname)"
echo "==> python:  $(which python3)  $(python3 --version)"
echo "==> CONFIG:  $CONFIG"
echo "==> OUTDIR:  $OUTDIR"
echo "==> start:   $(date -Iseconds)"

mkdir -p "$OUTDIR"

# Numba threading: hw2d uses Numba shared-memory accelerators. Match the
# threads to the node's cores. Frontera small queue: SKX=48, CLX=56.
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-${SLURM_CPUS_ON_NODE:-48}}"
echo "==> NUMBA_NUM_THREADS=$NUMBA_NUM_THREADS"

python3 -m hw.dns --config "$CONFIG" --out-dir "$OUTDIR"

echo "==> end:     $(date -Iseconds)"

# Quick post-run check (does not gate the job; full validate runs later).
python3 -m hw.crosscheck "$OUTDIR/trajectory.h5" || true
