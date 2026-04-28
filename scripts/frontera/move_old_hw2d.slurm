#!/bin/bash
#SBATCH -J move_old_hw2d
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o move_old_hw2d_%j.out
#SBATCH -e move_old_hw2d_%j.err
#
# Soft-delete: rsync $WORK/data/hw2d_sim -> $SCRATCH/archive/hw2d_sim_oldnu,
# remove source files on success. The archive will be auto-purged from
# $SCRATCH after ~10 days untouched.

set -euo pipefail

SRC="$WORK/data/hw2d_sim"
DST="$SCRATCH/archive/hw2d_sim_oldnu"

echo "==> start  $(date -Iseconds)  on $(hostname)"
echo "==> SRC:   $SRC"
echo "==> DST:   $DST"
du -sh "$SRC"
df -h "$SCRATCH" | tail -2

mkdir -p "$DST"

# rsync: -a preserve attrs, --remove-source-files drops src on success,
# --info=progress2 logs total progress.
rsync -av --remove-source-files --info=progress2 "$SRC/" "$DST/"

# rsync only removes files; tear down the now-empty dir tree.
find "$SRC" -type d -empty -delete

echo "==> end    $(date -Iseconds)"
echo "==> final state:"
du -sh "$DST"
if [ -d "$SRC" ]; then
    echo "WARN: $SRC still exists; contents:"
    ls -la "$SRC"
else
    echo "$SRC removed."
fi
