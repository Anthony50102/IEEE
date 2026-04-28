# Phase 1 on Frontera

One-time setup, then submit the four-alpha matrix. All DNS runs are
single-node, single-process (Numba shared memory; no MPI). Queue: `small`.

## 1. Setup Python (one time)

```bash
# from a Frontera login node, repo cloned to $WORK/IEEE
cd $WORK/IEEE
bash scripts/frontera/setup_python.sh
```

This uses `pip install --user --no-deps` so the system numpy / scipy /
h5py / matplotlib (with TACC's hardware-tuned BLAS) are NOT replaced.
Only pure-Python deps that are missing get installed into `~/.local`.

Verify:

```bash
python3 -c "import hw2d, numpy, scipy, h5py, numba, yaml; \
            print('numpy', numpy.__file__); print('hw2d ok')"
```

`numpy.__file__` should still point to a TACC system path, not
`~/.local/...`.

## 2. Smoke run on a development node (one time, ~1 min)

```bash
sbatch -p development -t 00:10:00 -J hw_smoke \
    --wrap "python3 -m hw.dns --config configs/data/hw_smoke.yaml \
            --out-dir \$SCRATCH/IEEE/data/hw2d/smoke && \
            python3 -m hw.crosscheck \$SCRATCH/IEEE/data/hw2d/smoke/trajectory.h5"
```

Crosscheck must say `PASS:`. If not, stop and fix.

## 3. Submit the alpha matrix

```bash
bash scripts/frontera/submit_all.sh
```

This submits four jobs, one per alpha (0.1, 1.0, 1.5, 5.0). Each writes
to `$SCRATCH/IEEE/data/hw2d/alpha<X>_n512/`. Time limit: 4h on `small`.
Trajectories are typically 5-15 GB each.

## 4. Validate

After all four complete:

```bash
python3 -m hw.validate \
    $SCRATCH/IEEE/data/hw2d/alpha*_n512/data_card.yaml
```

Expect verdict `ok` (z < 2) on all four. `warn` (2 <= z < 3) is
acceptable on the loosest configs. Any `fail` blocks Phase 2.
