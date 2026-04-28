#!/bin/bash
# setup_data.sh — Copy archived HW2D data from $WORK to $SCRATCH for fast Lustre I/O
#
# Usage:
#   bash scripts/setup_data.sh              # Copy ALL h5 files (use via SLURM, not login node)
#   bash scripts/setup_data.sh 3            # Copy only the first 3 h5 files (quick, ok on login node)
#   bash scripts/setup_data.sh <pattern>    # Copy files matching a glob pattern
#
# For full copies, submit as a SLURM job:
#   sbatch scripts/setup_data.slurm
#
# After running, update your YAML configs to point data_dir at:
#   $SCRATCH/IEEE/data/hw2d/

set -euo pipefail

ARCHIVE_DIR="$WORK/data/hw2d_sim/t600_d256x256_raw"
SCRATCH_DATA="$SCRATCH/IEEE/data/hw2d"
SCRATCH_OUTPUT="$SCRATCH/IEEE/output"

echo "=== Setting up SCRATCH data layout ==="

# Create directories
mkdir -p "$SCRATCH_DATA"
mkdir -p "$SCRATCH_OUTPUT"

# Set Lustre striping for parallel I/O (4 OSTs)
lfs setstripe -c 4 "$SCRATCH_DATA" 2>/dev/null || echo "Warning: could not set stripe count (run from compute node if needed)"

# Determine what to copy
if [ ! -d "$ARCHIVE_DIR" ]; then
    echo "ERROR: Archive directory not found: $ARCHIVE_DIR"
    exit 1
fi

if [ $# -eq 0 ]; then
    # Copy all files
    echo "Copying ALL HW2D data from $ARCHIVE_DIR..."
    cp -v "$ARCHIVE_DIR"/*.h5 "$SCRATCH_DATA/"
elif [[ "$1" =~ ^[0-9]+$ ]]; then
    # Copy first N files
    N=$1
    echo "Copying first $N HW2D files from $ARCHIVE_DIR..."
    ls "$ARCHIVE_DIR"/*.h5 | head -n "$N" | while read f; do
        cp -v "$f" "$SCRATCH_DATA/"
    done
else
    # Copy files matching pattern
    PATTERN="$1"
    echo "Copying HW2D files matching '$PATTERN' from $ARCHIVE_DIR..."
    cp -v "$ARCHIVE_DIR"/$PATTERN "$SCRATCH_DATA/"
fi

echo ""
echo "=== Setup complete ==="
echo "  Data:   $SCRATCH_DATA ($(ls "$SCRATCH_DATA"/*.h5 2>/dev/null | wc -l) files)"
echo "  Output: $SCRATCH_OUTPUT"
echo ""
echo "Update your YAML configs:"
echo "  data_dir: \"$SCRATCH_DATA\""
