#!/bin/bash
# Submit the four-alpha HW DNS matrix to Frontera.
#
# Run from the repo root on a login node:
#   cd $WORK/IEEE
#   bash scripts/frontera/submit_all.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

DATA_ROOT="${DATA_ROOT:-$WORK/data/IEEE/hw2d}"

CONFIGS=(
    "configs/data/hw_alpha0.1_n512.yaml"
    "configs/data/hw_alpha1.0_n512.yaml"
    "configs/data/hw_alpha1.5_n512.yaml"
    "configs/data/hw_alpha5.0_n512.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    label="$(basename "$cfg" .yaml)"          # e.g. hw_alpha1.0_n512
    outdir="$DATA_ROOT/${label#hw_}"          # -> $SCRATCH/.../alpha1.0_n512
    echo "==> submitting $label"
    sbatch \
        -J "$label" \
        --export=ALL,CONFIG="$cfg",OUTDIR="$outdir" \
        scripts/frontera/run_dns.slurm
done

echo "==> all submitted. check with: squeue -u \$USER"
echo "==> output root: $DATA_ROOT"
