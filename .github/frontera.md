# Frontera (TACC) Reference

Hard-won facts and the canonical workflow for running this paper's code on
TACC Frontera. Point new sessions / agents at this file before they touch
the cluster.

## Account / paths

| What             | Value                                           |
|------------------|-------------------------------------------------|
| Allocation       | `PHY25003` (only one; can omit `-A`)            |
| User             | `anthony50102`                                  |
| `$HOME`          | `/home1/10407/anthony50102` — 25 GB, backed up  |
| `$WORK`          | `/work2/10407/anthony50102/frontera` — 1 TB, NOT backed up |
| `$SCRATCH`       | `/scratch2/10407/anthony50102` — no quota, ~10-day purge of untouched files |
| Quota check      | `taccinfo`                                      |

`module list` shows quota lines on every login (`$HOME` 57%, `$WORK` 69%
historically — keep an eye if they climb).

## Filesystem layout for this project

```
$WORK/repos/IEEE/                  # code (this repo, refactor branch)
  hw/, rom/, configs/, scripts/, hw/hw2d/   (hw2d is a clone, gitignored)
$WORK/data/IEEE/hw2d/              # DNS trajectories (~50 GB; persistent)
  alpha0.1_n512/{trajectory.h5, data_card.yaml}
  alpha1.0_n512/, alpha1.5_n512/, alpha5.0_n512/
$WORK/results/IEEE/                # downstream artifacts (POD bases, ROMs, plots)
$SCRATCH/IEEE/jobs/                # SLURM logs, transient outputs
$SCRATCH/archive/                  # soft-deleted data (auto-purged ~10d)
```

**Why $WORK and not $SCRATCH for DNS data:** trajectories will be consumed
for weeks of surrogate training. $SCRATCH purges untouched files on a 10-day
clock; that risk is not worth the (free) $SCRATCH I/O. After cleaning, $WORK
has plenty of room.

## Python environment (CRITICAL)

Frontera's bashrc-injected `PYTHONPATH` and module quirks make this a
minefield. The single working recipe:

```bash
unset PYTHONPATH
module reset
module load python3/3.9.2 phdf5/1.10.4
```

This is in `scripts/frontera/env.sh`. **Source it (or use the wrapped
scripts) before any python3 / pip command.** What it gets you (all from
TACC system tree, hardware-tuned BLAS, do NOT replace):

- Python 3.9.2
- numpy 1.20.1, scipy 1.6.1
- h5py 3.2.1 (parallel-built, requires `phdf5` not `hdf5` module)
- matplotlib 3.3.4

### Gotchas

1. **`PYTHONPATH=/opt/apps/intel19/impi19_0/python3/3.7.0/lib/python3.7/site-packages`** is set somewhere in the user's startup. Until/unless that's removed from `~/.bashrc`, every script must `unset PYTHONPATH` first or 3.7-built `.so` files load into 3.9 and explode with `ImportError: libhdf5.so.103: cannot open shared object file` or `undefined symbol: H5Pget_fapl_mpio`.
2. **`module load hdf5/1.10.4`** is the *serial* HDF5 and gives `undefined symbol: H5Pget_fapl_mpio` because system h5py is parallel-built. Always use `phdf5/1.10.4`.
3. **No Python 3.10+** is available. Our code uses `from __future__ import annotations` so `int | None` etc. works under 3.9 (annotations stored as strings, never evaluated).
4. **Non-interactive SSH (e.g. `ssh frontera 'cmd'`) does not load the module function** — wrap with `bash -lc '...'` to get a login shell, or source `/etc/profile.d/zz_lmod.sh` manually.
5. **`pip install` policy:** ALWAYS use `--user --no-deps`. Anything else risks shadowing the BLAS-tuned numpy/scipy. `scripts/frontera/setup_python.sh` does this for the few pure-Python deps we need.

## Queues

| Queue        | Max walltime | Max nodes | Use for                              |
|--------------|--------------|-----------|--------------------------------------|
| `development`| 2 h          | 40        | smoke tests, interactive debugging   |
| `small`      | 48 h         | 2         | single-node jobs (our DNS, OpInf)    |
| `normal`     | 48 h         | 512       | multi-node training, parameter sweeps |
| `flex`       | 48 h         | varies    | low-priority preemptible             |

Our DNS is single-node single-process (Numba shared-memory, no MPI) →
`small` queue.

## Submitting

Standard SBATCH header for our work:

```bash
#SBATCH -J <jobname>
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o logs/<jobname>_%j.out
#SBATCH -e logs/<jobname>_%j.err
```

Account is omitted because `PHY25003` is the only one.

Inside the job body, FIRST line is always `source scripts/frontera/env.sh`.

`squeue -u $USER` for status. `scancel <jobid>` to kill. Logs go where
`-o` and `-e` say (relative to `$SLURM_SUBMIT_DIR`).

## Numba threading

hw2d uses Numba shared-memory accelerators. On `small` queue Skylake nodes
that's 48 cores, on Cascade Lake 56. Set `NUMBA_NUM_THREADS` to whatever
SLURM allocated:

```bash
export NUMBA_NUM_THREADS="${SLURM_CPUS_ON_NODE:-48}"
```

## Move/copy across filesystems

`$WORK` and `$SCRATCH` are different Lustre mounts; `mv` between them is
copy+delete, not rename. Use `rsync -av --info=progress2` and submit it as
a job (372 GB takes >10 min on a login node and would get killed).
Pattern in `scripts/frontera/move_old_hw2d.slurm`.

## Submit-host quota gate

Frontera's submit script blocks `sbatch` if any filesystem is at 99%+ usage
or > 95% file count. If sbatch fails with "Quota exceeded" check `taccinfo`
and rsync data to $SCRATCH/archive (or off-cluster) before retrying.

## SSH

`ssh $FRONTERA` (alias for `anthony50102@frontera.tacc.utexas.edu`) works
non-interactively (key + 2FA exemption already configured for this host).
For commands that need module/quota machinery: `ssh $FRONTERA 'bash -lc "..."'`.
