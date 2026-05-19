#!/bin/bash
#SBATCH -J disco_train
#SBATCH -o %x.o%j.log
#SBATCH -e %x.e%j.log
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00
#SBATCH -A PHY25003
#SBATCH --mail-type=fail,end

# Multi-α DISCO training on Vista GH200 (1 H200).
# See ROADMAP.md Phase 6 (p6-train) and knowledge/vista_compute.md.

set -euo pipefail

module reset
module load gcc/15.1.0 cuda/12.5 python3/3.11.8

source "$WORK/envs/disco/bin/activate"
cd "$WORK/repos/IEEE"

# --- output directory (outside the repo, under $SCRATCH) -----------------
ts="$(date +%Y%m%d_%H%M%S)"
# Disambiguate when multiple jobs start in the same second (we hit this
# with 709802+709803 colliding on 20260518_044221_disco_train and both
# writing to the same metrics.jsonl / checkpoints/ — recoverable but
# confusing).
out="$SCRATCH/IEEE/output/${ts}_${SLURM_JOB_ID}_disco_train"
mkdir -p "$out"
echo "out=$out"

# --- wandb -------------------------------------------------------------
# Compute nodes on Vista normally have outbound HTTPS, so wandb online
# works. If a run reports a connect error to wandb.ai, uncomment the
# next line and run `wandb sync $out/wandb/offline-run-*` from a login
# node afterward to upload.
# export WANDB_MODE=offline

# --- environment hygiene ----------------------------------------------
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
# Threads for the dataloader workers; H200 nodes have 72 Grace cores.
export OMP_NUM_THREADS=8

# --- run ----------------------------------------------------------------
# Config: defaults to hw2d_multi.yaml; override with
#   sbatch --export=ALL,CONFIG=configs/disco/hw2d_multi_tuned.yaml scripts/vista/disco_train.slurm
# To resume from a checkpoint, also set RESUME (absolute path), e.g.
#   sbatch --export=ALL,CONFIG=...,RESUME=$SCRATCH/IEEE/output/<ts>_disco_train/checkpoints/latest.pt scripts/vista/disco_train.slurm
# NOTE: with --export, prefix `ALL,` so the rest of the env (PATH, etc.)
# is inherited; bare `--export=FOO=bar` replaces the env and breaks
# module/venv setup.
CONFIG="${CONFIG:-configs/disco/hw2d_multi.yaml}"
RESUME="${RESUME:-}"
echo "config=$CONFIG"
[ -n "$RESUME" ] && echo "resume=$RESUME"

resume_args=()
if [ -n "$RESUME" ]; then
    resume_args=(--resume "$RESUME")
fi

# Add --wandb to opt into wandb.
srun python -m disco.train_hw2d \
    --config "$CONFIG" \
    --out "$out" \
    "${resume_args[@]}"

echo "done; artifacts in $out"
