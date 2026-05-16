# Vendored DISCO upstream

This directory is a vendored snapshot of:

- **Repository**: https://github.com/RudyMorel/DISCO
- **Commit**: `ddd18f170646c80ad665e425db71e6e6913e055d`
- **Date**: 2025-07-23
- **License**: MIT (see `LICENSE`)
- **Paper**: Morel, Han, Oyallon. "DISCO: learning to DISCover an evolution
  Operator for multi-physics-agnostic prediction." 2025. arXiv:2504.19496v1.

## What's here

- `models/` — `disco.py` (hypernetwork + operator), `attention.py` (transformer blocks).
- `torchdiffeq/` — authors' fork of `torchdiffeq` (adjoint ODE solver).
- `utils/` — `yparams.py` (YAML config loader), `time_tracker.py`,
  `logging_utils.py`, `standardize` (in `__init__.py`).
  **`data_utils.py` from upstream is *not* used** — we replace it with our own
  HW2D-specific dataloader at `IEEE/disco/hw2d_dataset.py`.
- `train_reference.py` — verbatim copy of upstream `train.py` for reference;
  our training entry point is `IEEE/disco/train_hw2d.py`.
- `config_reference/` — upstream's PDEBench/Well configs, kept as architectural
  reference for our `IEEE/configs/disco/` configs.

## What's NOT here

- `src/the_well/` — Well dataset loader; we use `hw.dataset` instead.
- `assets/` — figures; not needed.

## Modification policy

**Do not edit `upstream/`.** If a change is needed, copy the file out, edit it
in `IEEE/disco/`, and document the divergence here. This keeps the vendored
copy clean and lets us re-vendor a newer commit if needed.

## Re-vendor procedure

```bash
git clone --depth 1 https://github.com/RudyMorel/DISCO.git /tmp/DISCO_upstream
cp -r /tmp/DISCO_upstream/src/{models,torchdiffeq,utils} IEEE/disco/upstream/
cp /tmp/DISCO_upstream/LICENSE IEEE/disco/upstream/LICENSE
cp /tmp/DISCO_upstream/train.py IEEE/disco/upstream/train_reference.py
cp -r /tmp/DISCO_upstream/config IEEE/disco/upstream/config_reference
# update commit SHA in this file
```
